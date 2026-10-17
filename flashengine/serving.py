"""供外部服务调用的单线程引擎驱动；不实现网络服务或跨模型调度。"""

from dataclasses import dataclass, field
from queue import Queue, Empty
from threading import Event, Lock, Thread, BoundedSemaphore
from time import monotonic
import math


class ServiceUnavailable(RuntimeError):
    pass


@dataclass(eq=False)
class _Request:
    prompt: object
    sampling: object
    stream: bool
    callback: object
    cancelled: object
    deadline: float
    done: Event = field(default_factory=Event)
    result: dict | None = None
    error: Exception | None = None
    sequence: object = None
    offset: int = 0


class EngineService:
    """初始化、step、取消与 close 始终在同一线程执行；回调须及时返回。"""

    def __init__(self, factory, *, max_pending=1024, request_timeout=300,
                 startup_timeout=600, shutdown_timeout=30):
        if (isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending < 1
                or any(not math.isfinite(t) or t <= 0 for t in
                       (request_timeout, startup_timeout, shutdown_timeout))):
            raise ValueError("队列容量和超时必须为正数")
        self._factory = factory
        self._queue = Queue()
        self._slots = BoundedSemaphore(max_pending)
        self._lock = Lock()
        self._wake = Event()
        self._ready = Event()
        self._closing = False
        self._fatal = None
        self.request_timeout = request_timeout
        self.shutdown_timeout = shutdown_timeout
        self._thread = Thread(target=self._run, name="flashengine-driver", daemon=True)
        self._thread.start()
        if not self._ready.wait(startup_timeout):
            with self._lock:
                self._closing = True
            self._wake.set()
            raise TimeoutError("引擎初始化超时；驱动线程将在初始化返回后清理")
        if self._fatal is not None:
            self._thread.join(shutdown_timeout)
            raise ServiceUnavailable(f"引擎初始化失败：{self._fatal}") from self._fatal

    def submit(self, prompt, sampling, *, stream=False, callback=None, cancelled=None):
        with self._lock:
            if self._closing or self._fatal is not None:
                raise ServiceUnavailable("引擎不可用或正在关闭")
            if not self._slots.acquire(blocking=False):
                raise ServiceUnavailable("请求队列已满")
            request = _Request(prompt, sampling, stream, callback, cancelled,
                               monotonic() + self.request_timeout)
            self._queue.put(request)
        self._wake.set()
        return request

    def close(self):
        with self._lock:
            self._closing = True
        self._wake.set()
        self._thread.join(self.shutdown_timeout)
        if self._thread.is_alive():
            raise TimeoutError("引擎仍在执行；未跨线程强制释放设备资源")

    def _finish(self, request, result=None, error=None):
        if request.done.is_set():
            return
        request.result, request.error = result, error
        try:
            if request.callback:
                request.callback(result, error, True)
        except Exception:
            # 传输失败不污染模型执行状态；不重试 FINAL，避免重复完成。
            pass
        finally:
            request.done.set()
            self._slots.release()

    def _interruption(self, request):
        if self._closing:
            return ServiceUnavailable("引擎关闭，请求已取消")
        if monotonic() >= request.deadline:
            return TimeoutError("推理请求超时")
        if request.cancelled and request.cancelled():
            return InterruptedError("客户端已取消请求")
        return None

    def _run(self):
        engine = None
        active = []
        try:
            engine = self._factory()
            self._ready.set()
            while True:
                # 每轮只处理当前已入队的请求，保证持续入流时仍能推进 decode。
                for _ in range(self._queue.qsize()):
                    request = self._queue.get_nowait()
                    try:
                        interruption = self._interruption(request)
                        if interruption:
                            raise interruption
                        request.sequence = engine.add_request(request.prompt, request.sampling)
                        active.append(request)
                    except Exception as exc:
                        self._finish(request, error=exc)
                for request in active[:]:
                    try:
                        interruption = self._interruption(request)
                    except Exception as exc:
                        interruption = exc
                    if interruption:
                        engine.cancel_request(request.sequence)
                        active.remove(request)
                        self._finish(request, error=interruption)
                if self._closing and not active:
                    break
                if not active:
                    self._wake.wait(0.1)
                    self._wake.clear()
                    continue
                engine.step()
                for request in active[:]:
                    # kernel 返回后再检查，避免最后一步期间到期/取消却发成功 FINAL。
                    try:
                        interruption = self._interruption(request)
                    except Exception as exc:
                        interruption = exc
                    if interruption:
                        engine.cancel_request(request.sequence)
                        active.remove(request)
                        self._finish(request, error=interruption)
                        continue
                    seq = request.sequence
                    if not seq.is_finished and (not request.stream or seq.num_completion_tokens == request.offset):
                        continue
                    result = engine.request_snapshot(seq, request.offset if request.stream else 0)
                    # 文本返回累计值，避免 BPE 不完整字符使增量文本无法拼接。
                    result["text"] = engine.tokenizer.decode(seq.completion_token_ids)
                    request.offset = seq.num_completion_tokens
                    if seq.is_finished:
                        active.remove(request)
                        self._finish(request, result=result)
                    elif request.callback:
                        try:
                            request.callback(result, None, False)
                        except Exception:
                            engine.cancel_request(seq)
                            active.remove(request)
                            self._finish(request, error=ConnectionError("响应发送失败"))
        except Exception as exc:
            with self._lock:
                # 不保存设备异常的 traceback，避免其局部张量延长模型寿命。
                self._fatal = RuntimeError(f"{type(exc).__name__}: {exc}")
            for request in active:
                self._finish(request, error=ServiceUnavailable(f"引擎执行失败：{exc}"))
        finally:
            with self._lock:
                self._closing = True
            while True:
                try:
                    self._finish(self._queue.get_nowait(), error=ServiceUnavailable("引擎已停止"))
                except Empty:
                    break
            if engine is not None:
                try:
                    engine.close()
                except Exception as exc:
                    self._fatal = RuntimeError(f"{type(exc).__name__}: {exc}")
            self._ready.set()
