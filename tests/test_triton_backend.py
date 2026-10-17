"""用拥有相同接口的主机替身验证契约，不代表 Triton Server 实机验收。"""

from collections import deque
import json
from threading import Event, get_ident
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from flashengine.engine.requests import EngineRequests, RequestCapacityError
from flashengine.engine.sequence import SequenceStatus
from flashengine.sampling_params import SamplingParams
from flashengine.serving import EngineService, ServiceUnavailable
from flashengine.triton_backend import TritonPythonModel


class FakeEngine(EngineRequests):
    def __init__(self, delay=.002, fail=False):
        self.owner = get_ident()
        self.threads = [self.owner]
        self.delay, self.fail = delay, fail
        self._closed = False
        self.peak_active = 0
        self.prompts = []
        self.cancelled = []
        self.config = SimpleNamespace(model_config=SimpleNamespace(vocab_size=128),
                                      max_model_len=64, kvcache_block_size=4,
                                      max_pending_requests=128, eos=127)
        self.tokenizer = SimpleNamespace(encode=lambda s: [ord(c) % 128 for c in s],
                                         decode=lambda ids: " ".join(map(str, ids)))
        blocks = SimpleNamespace(blocks=list(range(32)), deallocate=lambda seq: seq.block_table.clear())
        self.scheduler = SimpleNamespace(waiting=deque(), running=deque(), block_manager=blocks)
        self.scheduler.add = self.scheduler.waiting.append

    def add_request(self, prompt, sampling):
        self.threads.append(get_ident())
        seq = super().add_request(prompt, sampling)
        self.prompts.append(list(seq.token_ids))
        return seq

    def _on_preempt_release(self, seq):
        self.cancelled.append(seq.seq_id)

    def step(self):
        self.threads.append(get_ident())
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("模拟设备执行失败")
        while self.scheduler.waiting and len(self.scheduler.running) < 2:
            seq = self.scheduler.waiting.popleft()
            self.scheduler.running.append(seq)
            seq.block_table.append(0)
        self.peak_active = max(self.peak_active, len(self.scheduler.running))
        for seq in list(self.scheduler.running):
            seq.append_token(1)
            if seq.num_completion_tokens >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                seq.block_table.clear()
                self.scheduler.running.remove(seq)

    def close(self):
        self.threads.append(get_ident())
        self._closed = True
        self.scheduler.waiting.clear()
        self.scheduler.running.clear()


class Tensor:
    def __init__(self, name, array):
        self.name, self.array = name, array

    def as_numpy(self):
        return self.array


class TritonError:
    CANCELLED, INVALID_ARG, UNAVAILABLE, INTERNAL = range(4)

    def __init__(self, message, code):
        self.message, self.code = message, code


class Response:
    def __init__(self, output_tensors=None, error=None):
        self.outputs = {t.name: t.array for t in output_tensors or []}
        self.error = error


class Sender:
    def __init__(self):
        self.responses = []
        self.finished = Event()
        self.cancelled = False

    def is_cancelled(self):
        return self.cancelled

    def send(self, response=None, flags=0):
        if self.finished.is_set():
            raise AssertionError("FINAL 后发送响应")
        self.responses.append((response, flags))
        if flags:
            self.finished.set()


class Request:
    def __init__(self, inputs=None, parameters=None):
        self.inputs = inputs if inputs is not None else {"PROMPT": np.array([b"Hi"], dtype=object)}
        self.params = parameters if parameters is not None else {"max_tokens": 3}
        self.sender = Sender()

    def parameters(self):
        return json.dumps(self.params)

    def get_response_sender(self):
        return self.sender

    def is_cancelled(self):
        return self.sender.cancelled


PB = SimpleNamespace(Tensor=Tensor, TritonError=TritonError, InferenceResponse=Response,
                     TRITONSERVER_RESPONSE_COMPLETE_FINAL=1,
                     get_input_tensor_by_name=lambda r, name: Tensor(name, r.inputs[name])
                     if name in r.inputs else None)


class TritonTests(unittest.TestCase):
    def test_gpu_instance_device_mapping_and_incompatible_runtime(self):
        config = {"max_batch_size": 0, "parameters": {
            "model_path": {"string_value": "/local/model"}}}
        args = {"model_config": json.dumps(config), "model_instance_kind": "GPU",
                "model_instance_device_id": "4"}
        observed = []
        adapter = TritonPythonModel()

        def factory(model, options):
            observed.append(options.copy())
            return FakeEngine()

        with patch.dict("sys.modules", {"triton_python_backend_utils": PB}), \
                patch("flashengine.backends.get_backend", return_value=SimpleNamespace(device_name="cuda")), \
                patch("flashengine.triton_backend._create_engine", side_effect=factory):
            adapter.initialize(args)
            adapter.finalize()
        self.assertEqual(observed[0]["device_ids"], [4])
        with patch.dict("sys.modules", {"triton_python_backend_utils": PB}), \
                patch("flashengine.backends.get_backend", return_value=SimpleNamespace(device_name="npu")), \
                patch("flashengine.triton_backend._create_engine") as create:
            with self.assertRaises(ServiceUnavailable):
                TritonPythonModel().initialize(args)
            create.assert_not_called()

    def model(self, decoupled=False, **options):
        model = TritonPythonModel()
        engines = []

        def factory(*args):
            engine = FakeEngine(**options)
            engines.append(engine)
            return engine

        config = {"max_batch_size": 0, "model_transaction_policy": {"decoupled": decoupled},
                  "parameters": {"model_path": {"string_value": "/local/model"},
                                 "engine_config": {"string_value": '{"device_ids":[0]}'}}}
        with patch.dict("sys.modules", {"triton_python_backend_utils": PB}), \
                patch("flashengine.triton_backend._create_engine", side_effect=factory):
            model.initialize({"model_config": json.dumps(config), "model_instance_kind": "CPU"})
        self.addCleanup(model.finalize)
        return model, engines[0]

    def test_nonstream_order_error_isolation_and_continuous_batching(self):
        model, engine = self.model()
        responses = model.execute([Request(), Request(inputs={}), Request()])
        self.assertEqual(len(responses), 3)
        self.assertEqual(responses[1].error.code, TritonError.INVALID_ARG)
        for response in (responses[0], responses[2]):
            self.assertIsNone(response.error)
            self.assertEqual(response.outputs["TOKEN_IDS"].tolist(), [1, 1, 1])
            self.assertEqual(response.outputs["FINISH_REASON"].tolist(), [b"length"])
        self.assertEqual(engine.peak_active, 2)
        model.finalize()
        self.assertEqual(set(engine.threads), {engine.owner})
        self.assertTrue(engine._closed)

    def test_decoupled_owns_input_and_sends_one_final(self):
        model, engine = self.model(decoupled=True, delay=.01)
        ids = np.array([4, 5], dtype=np.int64)
        request = Request(inputs={"INPUT_IDS": ids})
        self.assertIsNone(model.execute([request]))
        ids[:] = 99
        self.assertTrue(request.sender.finished.wait(3))
        self.assertEqual(engine.prompts, [[4, 5]])
        self.assertEqual([flags for _, flags in request.sender.responses], [0, 0, 1])
        tokens = [int(t) for response, _ in request.sender.responses for t in response.outputs["TOKEN_IDS"]]
        self.assertEqual(tokens, [1, 1, 1])

    def test_decoupled_nonstream_and_invalid_request(self):
        model, _ = self.model(decoupled=True)
        request = Request(inputs={"PROMPT": np.array([b"Hi"], dtype=object),
                                  "STREAM": np.array([False])})
        invalid = Request(inputs={"INPUT_IDS": np.array([1.1])})
        model.execute([request, invalid])
        self.assertTrue(request.sender.finished.wait(3))
        self.assertEqual(len(request.sender.responses), 1)
        self.assertEqual(invalid.sender.responses[0][0].error.code, TritonError.INVALID_ARG)
        self.assertTrue(invalid.sender.finished.is_set())

    def test_cancel_and_finalize_release_pending_requests(self):
        model, engine = self.model(decoupled=True, delay=.01)
        request = Request(parameters={"max_tokens": 40})
        model.execute([request])
        time.sleep(.03)
        request.sender.cancelled = True
        self.assertTrue(request.sender.finished.wait(3))
        self.assertEqual(request.sender.responses[-1][0].error.code, TritonError.CANCELLED)
        self.assertTrue(engine.cancelled)
        pending = Request(parameters={"max_tokens": 40})
        model.execute([pending])
        model.finalize()
        self.assertTrue(pending.sender.finished.is_set())
        self.assertTrue(engine._closed)
        model.finalize()

    def test_execution_failure_finishes_all_and_rejects_new(self):
        model, _ = self.model(decoupled=True, fail=True)
        requests = [Request(), Request()]
        model.execute(requests)
        for request in requests:
            self.assertTrue(request.sender.finished.wait(3))
            self.assertEqual(request.sender.responses[-1][0].error.code, TritonError.UNAVAILABLE)
        following = Request()
        model.execute([following])
        self.assertTrue(following.sender.finished.wait(3))

    def test_invalid_inputs_do_not_enter_engine(self):
        model, engine = self.model()
        invalid = [Request(parameters={"max_tokens": 0}),
                   Request(parameters={"temperature": float("nan")}),
                   Request(parameters={"max_tokens": 100}),
                   Request(inputs={"INPUT_IDS": np.array([-1], dtype=np.int64)}),
                   Request(inputs={"PROMPT": np.array([b"a", b"b"], dtype=object)}),
                   Request(inputs={"PROMPT": np.array([b"a"], dtype=object),
                                   "STREAM": np.array([True])})]
        responses = model.execute(invalid)
        self.assertTrue(all(r.error.code == TritonError.INVALID_ARG for r in responses))
        self.assertEqual(engine.prompts, [])

    def test_capacity_rejection_and_deadline_do_not_poison_engine(self):
        engines = []

        def factory():
            engine = FakeEngine(delay=.02)
            engines.append(engine)
            return engine

        service = EngineService(factory, max_pending=1, request_timeout=.05)
        self.addCleanup(service.close)
        job = service.submit("Hi", SamplingParams(max_tokens=40))
        with self.assertRaises(ServiceUnavailable):
            service.submit("Hi", SamplingParams(max_tokens=1))
        self.assertTrue(job.done.wait(3))
        self.assertIsInstance(job.error, TimeoutError)
        following = service.submit("Hi", SamplingParams(max_tokens=1))
        self.assertTrue(following.done.wait(3))
        self.assertIsNone(following.error)
        self.assertTrue(engines[0].cancelled)

    def test_initialization_failure_does_not_leave_driver(self):
        def fail():
            raise ValueError("模型无效")
        with self.assertRaises(ServiceUnavailable):
            EngineService(fail)

    def test_deadline_during_last_device_step_is_not_success(self):
        service = EngineService(lambda: FakeEngine(delay=.03), request_timeout=.01)
        self.addCleanup(service.close)
        finals = []
        job = service.submit("Hi", SamplingParams(max_tokens=1), stream=True,
                             callback=lambda result, error, final: finals.append((error, final)))
        self.assertTrue(job.done.wait(3))
        self.assertIsInstance(job.error, TimeoutError)
        self.assertEqual(len(finals), 1)
        self.assertTrue(finals[0][1])

    def test_public_cancel_partial_prefill_and_request_limits(self):
        engine = FakeEngine()
        seq = engine.add_request("Hi", SamplingParams(max_tokens=3))
        seq.block_table = [0]
        self.assertTrue(engine.cancel_request(seq.seq_id))
        self.assertFalse(engine.cancel_request(seq))
        self.assertEqual(engine.request_snapshot(seq)["finish_reason"], "cancelled")
        self.assertEqual(seq.block_table, [])
        engine.config.max_pending_requests = 1
        engine.add_request("Hi", SamplingParams(max_tokens=3))
        with self.assertRaises(RequestCapacityError):
            engine.add_request("Hi", SamplingParams(max_tokens=3))


if __name__ == "__main__":
    unittest.main()
