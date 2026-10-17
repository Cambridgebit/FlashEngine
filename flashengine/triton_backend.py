"""外部 Triton Python Backend 可直接导入的薄适配入口。"""

import importlib
import json
import math
import os
from dataclasses import fields
from threading import Event
from time import monotonic, sleep

import numpy as np

from flashengine.sampling_params import SamplingParams
from flashengine.serving import EngineService, ServiceUnavailable
from flashengine.engine.requests import RequestCapacityError


def _create_engine(model, options):
    from flashengine.config import Config
    from flashengine import LLM
    allowed = {f.name for f in fields(Config) if f.init} - {"model", "hf_config", "shm_name"}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"未知引擎参数：{sorted(unknown)}")
    return LLM(model, **options)


class TritonPythonModel:
    def initialize(self, args):
        self._pb = importlib.import_module("triton_python_backend_utils")
        config = json.loads(args["model_config"])
        if config.get("max_batch_size", 0) != 0:
            raise ValueError("请设置 max_batch_size: 0；请求由引擎连续批处理")
        self._decoupled = bool(config.get("model_transaction_policy", {}).get("decoupled", False))
        parameters = config.get("parameters", {})

        def parameter(name, default=None):
            return parameters.get(name, {}).get("string_value", default)

        model = parameter("model_path")
        if not model:
            raise ValueError("parameters.model_path 必须指定本地模型目录")
        options = json.loads(parameter("engine_config", "{}"))
        if not isinstance(options, dict):
            raise ValueError("engine_config 必须为 JSON 对象")
        kind = args.get("model_instance_kind", "CPU").removeprefix("KIND_")
        if kind == "GPU":
            device_id = int(args["model_instance_device_id"])
            if options.get("tensor_parallel_size", 1) != 1:
                raise ValueError("GPU 实例限 TP=1；多卡 TP 请使用 CPU 实例并显式指定 device_ids")
            # GPU placement 只映射 CUDA 兼容设备；NPU 必须显式选设备。
            if "device_ids" in options and tuple(options["device_ids"]) != (device_id,):
                raise ValueError("device_ids 与 Triton GPU 实例设备冲突")
            options["device_ids"] = [device_id]
        elif kind == "CPU":
            if not options.get("device_ids"):
                raise ValueError("CPU 实例必须显式配置引擎 device_ids")
        else:
            raise ValueError(f"不支持实例类型：{kind}")
        timeouts = {}
        for key, default in (("request_timeout", 300), ("startup_timeout", 600),
                             ("shutdown_timeout", 30)):
            value = float(parameter(key, str(default)))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} 必须为正的有限秒数")
            timeouts[key] = value
        def factory():
            if kind == "GPU":
                from flashengine.backends import get_backend
                if options.get("backend", "auto") != "auto":
                    os.environ["FLASHENGINE_BACKEND"] = options["backend"]
                if get_backend().device_name != "cuda":
                    raise ValueError("该设备运行时必须使用 CPU 实例及显式 device_ids")
            return _create_engine(model, options)

        self._service = EngineService(factory,
                                      max_pending=int(options.get("max_pending_requests", 1024)),
                                      **timeouts)

    def _input(self, request, name):
        tensor = self._pb.get_input_tensor_by_name(request, name)
        return None if tensor is None else tensor.as_numpy()

    def _parse(self, request):
        text, ids = self._input(request, "PROMPT"), self._input(request, "INPUT_IDS")
        if (text is None) == (ids is None):
            raise ValueError("PROMPT 与 INPUT_IDS 必须且只能提供一个")
        if text is not None:
            if text.shape != (1,) or not isinstance(text[0], (str, bytes)):
                raise ValueError("PROMPT 必须包含一个 UTF-8 字符串")
            prompt = text[0].decode("utf-8") if isinstance(text[0], bytes) else str(text[0])
        else:
            if ids.ndim != 1 or ids.size == 0 or ids.dtype.kind not in "iu":
                raise ValueError("INPUT_IDS 必须为非空的一维整数张量")
            prompt = ids.tolist()
        raw = json.loads(request.parameters() or "{}")
        allowed = {f.name for f in fields(SamplingParams)}
        if set(raw) - allowed:
            raise ValueError(f"未知采样参数：{sorted(set(raw) - allowed)}")
        sampling = SamplingParams(**{"temperature": 0.0, **raw})
        stream = self._input(request, "STREAM")
        if stream is not None and (stream.shape != (1,) or stream.dtype.kind != "b"):
            raise ValueError("STREAM 必须为单元素布尔张量")
        stream = bool(stream[0]) if stream is not None else self._decoupled
        if stream and not self._decoupled:
            raise ValueError("流式请求需要 decoupled 模型配置")
        return prompt, sampling, stream

    def _response(self, result=None, error=None):
        pb = self._pb
        if error is not None:
            code = (pb.TritonError.CANCELLED if isinstance(error, InterruptedError)
                    else pb.TritonError.INVALID_ARG if isinstance(error, (ValueError, AssertionError, TypeError))
                    else pb.TritonError.UNAVAILABLE if isinstance(error, (ServiceUnavailable, RequestCapacityError))
                    else pb.TritonError.INTERNAL)
            return pb.InferenceResponse(error=pb.TritonError(str(error), code))
        values = [("TOKEN_IDS", result["token_ids"], np.int64),
                  ("TEXT", [result["text"].encode("utf-8")], object),
                  ("FINISH_REASON", [result["finish_reason"].encode("utf-8")], object),
                  ("PROMPT_TOKENS", [result["prompt_tokens"]], np.int64),
                  ("COMPLETION_TOKENS", [result["completion_tokens"]], np.int64)]
        return pb.InferenceResponse(output_tensors=[pb.Tensor(name, np.array(value, dtype=dtype))
                                                    for name, value, dtype in values])

    def execute(self, requests):
        pending = []
        for request in requests:
            sender = request.get_response_sender() if self._decoupled else None
            try:
                # 转为拥有独立存储的 str/list 后，异步路径不保留 request 或输入视图。
                prompt, sampling, stream = self._parse(request)
                if sender is not None:
                    def send(result, error, final, sender=sender):
                        flags = self._pb.TRITONSERVER_RESPONSE_COMPLETE_FINAL if final else 0
                        sender.send(self._response(result, error), flags=flags)
                    self._service.submit(prompt, sampling, stream=stream, callback=send,
                                         cancelled=sender.is_cancelled)
                else:
                    cancelled = Event()
                    job = self._service.submit(prompt, sampling, cancelled=cancelled.is_set)
                    pending.append((job, request, cancelled))
            except Exception as exc:
                response = self._response(error=exc)
                if sender is not None:
                    sender.send(response, flags=self._pb.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                else:
                    pending.append(response)
        if self._decoupled:
            return None
        responses = [None] * len(pending)
        while any(response is None for response in responses):
            for i, item in enumerate(pending):
                if responses[i] is not None:
                    continue
                if not isinstance(item, tuple):
                    responses[i] = item
                    continue
                job, request, cancelled = item
                if job.done.is_set():
                    responses[i] = self._response(job.result, job.error)
                elif request.is_cancelled():
                    cancelled.set()
                    responses[i] = self._response(error=InterruptedError("客户端已取消请求"))
                elif monotonic() >= job.deadline:
                    cancelled.set()
                    responses[i] = self._response(error=TimeoutError("推理请求超时"))
            if any(response is None for response in responses):
                sleep(0.01)
        return responses

    def finalize(self):
        service = getattr(self, "_service", None)
        if service is not None:
            service.close()
