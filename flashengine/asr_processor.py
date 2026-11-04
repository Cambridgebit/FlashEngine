"""Qwen3-ASR 音频输入管线：log-mel、wav 读取、prompt 构造与输出解析。

对齐 transformers 5.15 的 `Qwen3ASRFeatureExtractor` / `Qwen3ASRProcessor`：`log_mel`
与参考实现逐位一致（fp32），prompt 复用模型自带的 tokenizer chat template 生成。

mel 口径（transformers 5.15 models/qwen3_asr/feature_extraction_qwen3_asr.py）：

- mel filter：`mel_filter_bank(1 + n_fft // 2, 128, 0.0, 8000.0, 16000, norm="slaney",
  mel_scale="slaney")`；
- `torch.stft(x, n_fft=400, hop=160, window=hann_window(400))`（periodic 窗），丢弃最后一帧；
- `|stft|^2` → `mel_filters.T @ magnitudes` → `clamp(1e-10).log10()` →
  `max(log_spec, log_spec.max() - 8)` → `(log_spec + 4) / 4`；
- 时间轴右侧补零到 `2 * n_window = 100` 的整数倍；有效帧数 = `L // hop_length`。
"""

from __future__ import annotations

import json
import math
import os
import struct
from typing import Any

import numpy as np

# 特殊 token（与 tokenizer_config.json / config.json 一致）
AUDIO_PAD_ID = 151676
AUDIO_START_ID = 151669
AUDIO_END_ID = 151670
ASR_TEXT_TOKEN = "<asr_text>"
IM_START_ID = 151644
IM_END_ID = 151645

# 特殊标记（<|im_end|> / <|im_start|> / <|audio_pad|> …）——解析输出前统一剥掉。

# 官方默认 system prompt 为空串（用空 system 段）
DEFAULT_SYSTEM_PROMPT = ""

# 语种 code → 官方全名（与 Qwen3ASRProcessor.LANGUAGE_CODE_TO_NAME 一致）
LANGUAGE_CODE_TO_NAME = {
    "ar": "Arabic",
    "yue": "Cantonese",
    "zh": "Chinese",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "fil": "Filipino",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "es": "Spanish",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
}

# 音频编码器默认超参（来自 preprocessor_config.json，两者需保持一致）
DEFAULT_SAMPLING_RATE = 16000
DEFAULT_N_FFT = 400
DEFAULT_HOP_LENGTH = 160
DEFAULT_N_MEL_BINS = 128
DEFAULT_N_WINDOW = 50
DEFAULT_MIN_LENGTH = 8000
DEFAULT_MAX_SAMPLES = 480000


# ---------------------------------------------------------------------------
# mel filter bank（numpy 复刻 transformers.audio_utils.mel_filter_bank）
# ---------------------------------------------------------------------------


def _hertz_to_mel(freq, mel_scale: str = "slaney"):
    """Hz → mel（slaney 尺度，Qwen3-ASR 用这一个分支）。"""
    if mel_scale == "slaney":
        min_log_hertz = 1000.0
        min_log_mel = 15.0
        logstep = 27.0 / np.log(6.4)
        mels = 3.0 * freq / 200.0
        if isinstance(freq, np.ndarray):
            log_region = freq >= min_log_hertz
            mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
        elif freq >= min_log_hertz:
            mels = min_log_mel + np.log(freq / min_log_hertz) * logstep
        return mels
    raise ValueError(f"只支持 slaney mel 尺度，收到 {mel_scale!r}")


def _mel_to_hertz(mels, mel_scale: str = "slaney"):
    """mel → Hz（slaney 尺度）。"""
    if mel_scale == "slaney":
        min_log_hertz = 1000.0
        min_log_mel = 15.0
        logstep = np.log(6.4) / 27.0
        freq = 200.0 * mels / 3.0
        log_region = mels >= min_log_mel
        freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
        return freq
    raise ValueError(f"只支持 slaney mel 尺度，收到 {mel_scale!r}")


def mel_filter_bank(
    num_frequency_bins: int,
    num_mel_filters: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int,
) -> np.ndarray:
    """slaney 归一化的三角 mel 滤波器组，返回 [num_frequency_bins, num_mel_filters] float64。"""
    mel_min = _hertz_to_mel(np.float64(min_frequency))
    mel_max = _hertz_to_mel(np.float64(max_frequency))
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = _mel_to_hertz(mel_freqs)
    fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)

    filter_diff = np.diff(filter_freqs)
    slopes = filter_freqs[None, :] - fft_freqs[:, None]
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    mel_filters = np.maximum(0.0, np.minimum(down_slopes, up_slopes))

    # slaney 面积归一化：让每个通道能量近似恒定
    enorm = 2.0 / (filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters])
    mel_filters *= enorm[None, :]
    return mel_filters


def _log_mel_torch(
    waveform: np.ndarray,
    mel_filters: np.ndarray,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    """torch 版 log-mel：与 HF `_torch_extract_fbank_features` 逐算子一致（fp32 位等）。"""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))
    window = torch.hann_window(n_fft)
    stft = torch.stft(x, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = (stft[..., :-1].abs() ** 2).contiguous()  # 丢弃最后一帧
    filters = torch.from_numpy(np.ascontiguousarray(mel_filters, dtype=np.float32))
    mel_spec = filters.T @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.numpy().astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# wav 读取 / 重采样
# ---------------------------------------------------------------------------


def _parse_riff_wav(data: bytes, source: str = "<memory>") -> tuple[np.ndarray, int]:
    """最小 RIFF/WAVE 解析：支持 PCM 8/16/24/32bit、IEEE float32/64、extensible。

    返回 (float32 单声道 [-1, 1] 采样, 采样率)。多声道按算术平均混成单声道。
    不用标准库 `wave` 是因为它不支持 24-bit/float 编码（本仓 asr_en.wav 是 24-bit）。
    """
    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"不是 RIFF/WAVE 文件：{source}")

    pos = 12
    fmt = None
    payload = None
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        (chunk_size,) = struct.unpack("<I", data[pos + 4 : pos + 8])
        body = data[pos + 8 : pos + 8 + chunk_size]
        if chunk_id == b"fmt ":
            fmt = body
        elif chunk_id == b"data":
            payload = body
        pos += 8 + chunk_size + (chunk_size & 1)  # RIFF 块按偶数字节对齐

    if fmt is None or payload is None:
        raise ValueError(f"wav 缺少 fmt/data 块：{source}")

    audio_format, num_channels, sample_rate = struct.unpack("<HHI", fmt[0:8])
    bits_per_sample = struct.unpack("<H", fmt[14:16])[0]
    if audio_format == 0xFFFE and len(fmt) >= 26:
        # WAVE_FORMAT_EXTENSIBLE：真实格式在 SubFormat GUID 的前 2 字节
        (audio_format,) = struct.unpack("<H", fmt[24:26])

    width = bits_per_sample // 8
    if audio_format == 3 and bits_per_sample == 32:
        samples = np.frombuffer(payload, dtype="<f4").astype(np.float32)
    elif audio_format == 3 and bits_per_sample == 64:
        samples = np.frombuffer(payload, dtype="<f8").astype(np.float32)
    elif audio_format == 1 and bits_per_sample == 8:
        samples = (np.frombuffer(payload, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif audio_format == 1 and bits_per_sample == 16:
        samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    elif audio_format == 1 and bits_per_sample == 24:
        raw = np.frombuffer(payload[: (len(payload) // 3) * 3], dtype=np.uint8).reshape(-1, 3)
        value = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        # 24-bit 补码符号扩展
        value = np.where(value >= 1 << 23, value - (1 << 24), value)
        samples = value.astype(np.float32) / float(1 << 23)
    elif audio_format == 1 and bits_per_sample == 32:
        samples = np.frombuffer(payload, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(
            f"不支持的 wav 编码：audio_format={audio_format}, bits={bits_per_sample}, source={source}"
        )

    if num_channels > 1:
        samples = samples[: (samples.shape[0] // num_channels) * num_channels]
        samples = samples.reshape(-1, num_channels).mean(axis=1)
    assert width > 0
    return np.ascontiguousarray(samples, dtype=np.float32), int(sample_rate)


def _read_riff_wav(path: str) -> tuple[np.ndarray, int]:
    """读 wav 文件 → (float32 单声道采样, 采样率)。"""
    with open(path, "rb") as f:
        data = f.read()
    return _parse_riff_wav(data, source=str(path))


def _design_lowpass_filter(num_taps: int, cutoff: float) -> np.ndarray:
    """Blackman 窗 sinc 低通；`cutoff` 为归一化截止频率（cycles/sample）。"""
    n = np.arange(num_taps, dtype=np.float64) - (num_taps - 1) / 2.0
    kernel = 2.0 * cutoff * np.sinc(2.0 * cutoff * n)
    kernel = kernel * np.blackman(num_taps)
    total = float(kernel.sum())
    if total == 0.0:
        raise ValueError("低通滤波器设计失败：窗函数权重之和为 0")
    return kernel / total


def resample_audio(
    waveform: np.ndarray, src_rate: int, dst_rate: int = DEFAULT_SAMPLING_RATE
) -> np.ndarray:
    """带抗混叠低通的有理数多相重采样（纯 numpy，无第三方依赖）。

    为什么不是裸线性插值：48k→16k 属于 3 倍抽取，直接抽点会把 8kHz 以上的能量折叠
    回带内（实测 mel 与标准重采样差 0.1 量级，见 docs/QWEN3_ASR_REFERENCE.md）。官方
    推理侧（librosa/soxr、torchaudio）都是「先低通再抽取」，这里用 Blackman 窗 sinc
    低通 + 多相结构复刻同一口径：上采样 L、下采样 M 由 gcd 归约，截止频率取两速率
    Nyquist 的较小者；滤波器直接按相位表在输入信号上加权（不实际零插值，内存占用
    与 44.1k→16k 这类大 L 也稳定）。

    输出长度 = ceil(n * dst_rate / src_rate)，与 torchaudio / librosa 一致。
    """
    if src_rate == dst_rate:
        return np.ascontiguousarray(waveform, dtype=np.float32).reshape(-1)
    src_rate = int(src_rate)
    dst_rate = int(dst_rate)
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError(f"采样率必须为正：src={src_rate}, dst={dst_rate}")
    x = np.ascontiguousarray(waveform, dtype=np.float32).reshape(-1)
    n = x.shape[0]
    if n == 0:
        return x
    out_len = int(math.ceil(n * float(dst_rate) / float(src_rate)))
    if out_len <= 0:
        return np.zeros(0, dtype=np.float32)

    g = math.gcd(src_rate, dst_rate)
    up = dst_rate // g    # 上采样因子 L
    down = src_rate // g  # 下采样因子 M
    fs_up = float(src_rate) * up
    # 归一化截止频率：通带上限取两速率中较低的 Nyquist
    cutoff = 0.5 * float(min(src_rate, dst_rate)) / fs_up
    half = int(math.ceil(5.0 / cutoff))
    num_taps = 2 * half + 1
    kernel = _design_lowpass_filter(num_taps, cutoff)
    taps_per_phase = (num_taps + up - 1) // up

    # 两侧补零，保证 base - k 落在合法区间（边界按零处理，与 librosa 一致）
    padded = np.concatenate(
        [
            np.zeros(taps_per_phase, dtype=np.float64),
            x.astype(np.float64),
            np.zeros(taps_per_phase + down + 1, dtype=np.float64),
        ]
    )
    offsets = np.arange(taps_per_phase, dtype=np.int64)
    # 零相位（线性相位）滤波器的群延迟：卷积后第 τ 点对应输入时刻 τ - delay，
    # 所以取输出 m 时必须从 τ = m*down + delay 处取样，否则整段波形会提前 delay/up 个
    # 输入样本（48k→16k 下约 10 个输出样本 = 0.6ms，足以让 mel 差 0.1 量级）。
    delay = (num_taps - 1) // 2
    out = np.empty(out_len, dtype=np.float32)
    chunk = 1 << 15
    for start in range(0, out_len, chunk):
        stop = min(out_len, start + chunk)
        t = np.arange(start, stop, dtype=np.int64) * down + delay
        phase = t % up
        base = t // up
        tap_index = phase[:, None] + offsets[None, :] * up  # [rows, taps_per_phase]
        weights = np.where(tap_index < num_taps, kernel[np.minimum(tap_index, num_taps - 1)], 0.0)
        gather = base[:, None] - offsets[None, :] + taps_per_phase
        gathered = padded[np.clip(gather, 0, padded.shape[0] - 1)]
        out[start:stop] = (gathered * weights).sum(axis=1).astype(np.float32)
    return np.ascontiguousarray(out)


def load_audio(path: str, target_sampling_rate: int = DEFAULT_SAMPLING_RATE) -> np.ndarray:
    """读 wav 并转成目标采样率的 float32 单声道波形（仓库唯一的重采样入口）。"""
    samples, rate = _read_riff_wav(path)
    return resample_audio(samples, rate, target_sampling_rate)


def decode_wav_bytes(data: bytes, target_sampling_rate: int = DEFAULT_SAMPLING_RATE) -> np.ndarray:
    """从内存里的 wav 字节解出 float32 单声道波形（HTTP 上传 / 压测脚本用）。

    服务端与 benchmark 应当走这里，而不是各自实现一遍「解码 + 线性重采样」——否则
    CER 参考口径与引擎实际喂给模型的波形不一致（`scripts/asr_benchmark.py` 目前
    仍是自己那套 np.interp，需要 asr-ops 改成调用本函数）。
    """
    samples, rate = _parse_riff_wav(data, source="<memory>")
    return resample_audio(samples, rate, target_sampling_rate)


def resolve_language(language: str | None) -> str | None:
    """把语种 code 或全名归一化成官方全名（None 表示自动检测）。"""
    if language is None:
        return None
    if not isinstance(language, str):
        raise TypeError(f"language 必须是字符串或 None，收到 {type(language)!r}")
    key = language.strip()
    if not key:
        return None
    code = LANGUAGE_CODE_TO_NAME.get(key.lower())
    if code is not None:
        return code
    lowered = key.lower()
    for name in LANGUAGE_CODE_TO_NAME.values():
        if name.lower() == lowered:
            return name
    raise ValueError(
        f"不支持的语种 {language!r}；可用 code：{sorted(LANGUAGE_CODE_TO_NAME)}，或使用全名"
    )


# ---------------------------------------------------------------------------
# AsrProcessor
# ---------------------------------------------------------------------------


class AsrProcessor:
    """Qwen3-ASR 的音频/文本前处理（对齐 HF Qwen3ASRProcessor 口径）。"""

    def __init__(
        self,
        tokenizer,
        chat_template: str,
        *,
        model_path: str | None = None,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        n_mel_bins: int = DEFAULT_N_MEL_BINS,
        n_window: int = DEFAULT_N_WINDOW,
        sampling_rate: int = DEFAULT_SAMPLING_RATE,
        min_length: int = DEFAULT_MIN_LENGTH,
    ) -> None:
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.model_path = model_path
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_mel_bins = int(n_mel_bins)
        self.n_window = int(n_window)
        self.sampling_rate = int(sampling_rate)
        self.min_length = int(min_length)
        self.chunk_len = 2 * self.n_window  # 100 帧一个 chunk
        self.audio_pad_id = AUDIO_PAD_ID
        self.audio_start_id = AUDIO_START_ID
        self.audio_end_id = AUDIO_END_ID
        # float64 计算后转 float32，与参考实现 `mel_filter_bank(...)` → `.to(float32)` 一致
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + self.n_fft // 2,
            num_mel_filters=self.n_mel_bins,
            min_frequency=0.0,
            max_frequency=float(self.sampling_rate // 2),
            sampling_rate=self.sampling_rate,
        ).astype(np.float32)

    # -- 构造 ---------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_path: str) -> "AsrProcessor":
        """从模型目录构造：读 preprocessor_config.json + tokenizer（含 chat template）。"""
        from transformers import AutoTokenizer

        preprocessor_path = os.path.join(model_path, "preprocessor_config.json")
        preprocessor: dict[str, Any] = {}
        if os.path.isfile(preprocessor_path):
            with open(preprocessor_path, "r", encoding="utf-8") as f:
                preprocessor = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        chat_template = _load_chat_template(model_path, tokenizer)
        # 参考实现里 tokenizer.chat_template 是 None（模板只挂在 processor 上），
        # 这里补上以便 apply_chat_template 可用。
        if getattr(tokenizer, "chat_template", None) is None:
            tokenizer.chat_template = chat_template

        return cls(
            tokenizer,
            chat_template,
            model_path=model_path,
            n_fft=int(preprocessor.get("n_fft", DEFAULT_N_FFT)),
            hop_length=int(preprocessor.get("hop_length", DEFAULT_HOP_LENGTH)),
            n_mel_bins=int(preprocessor.get("feature_size", DEFAULT_N_MEL_BINS)),
            n_window=int(preprocessor.get("n_window", DEFAULT_N_WINDOW)),
            sampling_rate=int(preprocessor.get("sampling_rate", DEFAULT_SAMPLING_RATE)),
        )

    # -- log-mel ------------------------------------------------------------

    def log_mel(self, waveform, sampling_rate: int = DEFAULT_SAMPLING_RATE) -> tuple[np.ndarray, int]:
        """波形 → (mel[128, T_pad], valid_frames)。

        Args:
            waveform: 单声道波形，np.ndarray 或 torch.Tensor 均可。
            sampling_rate: 输入波形的采样率；不是 16kHz 时按多相 FIR 抗混叠重采样。

        T_pad 是 valid_frames 向上补零到 `2 * n_window = 100` 的整数倍；valid_frames
        等于参考实现 attention_mask 的 1 的个数（= len(waveform) // hop_length）。
        """
        if hasattr(waveform, "detach"):  # torch.Tensor
            waveform = waveform.detach().cpu().numpy()
        x = np.ascontiguousarray(np.asarray(waveform, dtype=np.float32).reshape(-1))
        if x.shape[0] == 0:
            raise ValueError("音频长度不能为 0")
        if int(sampling_rate) != self.sampling_rate:
            x = resample_audio(x, int(sampling_rate), self.sampling_rate)
        if x.shape[0] < self.min_length:
            # 与参考实现 min_length 语义一致：短音频右侧补零到 8000 样本（16kHz 0.5s）
            x = np.pad(x, (0, self.min_length - x.shape[0]))

        log_spec = _log_mel_torch(x, self.mel_filters, self.n_fft, self.hop_length)
        valid_frames = log_spec.shape[1]

        remainder = valid_frames % self.chunk_len
        pad = (self.chunk_len - remainder) % self.chunk_len
        if pad:
            log_spec = np.pad(log_spec, ((0, 0), (0, pad)))
        return np.ascontiguousarray(log_spec, dtype=np.float32), valid_frames

    # -- token 数 -----------------------------------------------------------

    def num_audio_tokens(self, valid_frames: int) -> int:
        """mel 有效帧数 → 音频 token 数（三次 conv 的 stride=2 下采样）。"""
        valid_frames = int(valid_frames)
        if valid_frames < 0:
            raise ValueError(f"valid_frames 不能为负：{valid_frames}")
        remainder = valid_frames % self.chunk_len
        feat_lengths = (remainder - 1) // 2 + 1  # 第一次 conv（stride=2）
        per_chunk_tokens = (feat_lengths - 1) // 2 + 1  # 第二次 conv
        token_lengths = (per_chunk_tokens - 1) // 2 + 1  # 第三次 conv
        return int(token_lengths + (valid_frames // self.chunk_len) * 13)

    # -- prompt -------------------------------------------------------------

    def _render_prompt_ids(self, system_prompt: str | None, language: str | None) -> list[int]:
        """渲染 system + user + assistant 前缀的 input_ids（单个 audio_pad 占位）。"""
        messages: list[dict[str, Any]] = []
        system_text = DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt
        if system_text:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_text}]})
        messages.append({"role": "user", "content": [{"type": "audio"}]})

        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
        )
        input_ids = list(rendered["input_ids"])
        # 参考实现用 continue_final_message=True 把 assistant 前缀接在生成提示后面；
        # 但本 checkpoint 的 chat template 不渲染 assistant 段（HF 自己会抛
        # "final message does not appear in the chat"），故这里等价地手工追加前缀。
        prefix = f"language {language}{ASR_TEXT_TOKEN}" if language else ""
        if prefix:
            input_ids.extend(self.tokenizer.encode(prefix, add_special_tokens=False))
        return input_ids

    def _expand_audio_pad(self, input_ids: list[int], num_tokens: int) -> tuple[list[int], tuple[int, int]]:
        """把单个 audio_pad 展开成 num_tokens 个，返回 (ids, audio_span)。"""
        try:
            pos = input_ids.index(self.audio_pad_id)
        except ValueError as exc:
            raise ValueError("chat template 渲染结果里找不到 <|audio_pad|> 占位") from exc
        ids = input_ids[:pos] + [self.audio_pad_id] * num_tokens + input_ids[pos + 1 :]
        return ids, (pos, pos + num_tokens)

    def build(
        self,
        audio: str | np.ndarray,
        language: str | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """构造一次转写请求的全部输入。

        Args:
            audio: wav 路径，或已是 16kHz 单声道 float32 的波形数组。
            language: 语种 code/全名（如 "en"/"English"）；None 表示让模型自动检测。
            system_prompt: 可选 system 提示（热词/上下文）；None 用官方默认空串。

        Returns:
            dict：input_ids / audio_span / num_audio_tokens / mel / valid_frames / language。
        """
        if isinstance(audio, (str, os.PathLike)):
            waveform = load_audio(os.fspath(audio), self.sampling_rate)
        else:
            waveform = np.asarray(audio, dtype=np.float32).reshape(-1)

        language_name = resolve_language(language)
        mel, valid_frames = self.log_mel(waveform)
        num_tokens = self.num_audio_tokens(valid_frames)
        prompt_ids = self._render_prompt_ids(system_prompt, language_name)
        input_ids, audio_span = self._expand_audio_pad(prompt_ids, num_tokens)
        return {
            "input_ids": input_ids,
            "audio_span": audio_span,
            "num_audio_tokens": num_tokens,
            "mel": mel,
            "valid_frames": valid_frames,
            "language": language_name,
        }

    # -- 输出解析 -----------------------------------------------------------

    def parse(self, text: str, language: str | None = None) -> dict[str, Any]:
        """解析模型输出文本：`language <English><asr_text>正文` → {language, transcription}。

        参数
        ----
        text:
            引擎/参考实现解码出来的原始文本（可能带 `<|im_end|>` 之类的特殊标记）。
        language:
            请求时强制指定的语种。强制语种时该前缀是作为 assistant 起始喂给模型的，
            模型直接吐正文、不会回显 `language <X><asr_text>`，此时用本参数回填结果里的
            language（输出里若真带前缀，则以前缀解析出来的为准）。

        调用方约定：`scripts/asr_benchmark.py` / `scripts/openai_server.py` 在请求带
        language 时必须传 `parse(text, language=requested)`，否则被强制的语种会丢。

        兜底行为：先按 `<|...|>` 正则剥掉所有特殊标记（`<|im_end|>` 等）；入参为
        None / 空白串时返回 `{"language": language, "transcription": ""}`。
        """
        if not text:
            return {"language": language, "transcription": ""}

        cleaned = text.strip()
        if ASR_TEXT_TOKEN not in cleaned:
            detected = _normalize_language_prefix(cleaned)
            return {"language": detected or language, "transcription": cleaned.strip()}

        prefix, transcription = cleaned.split(ASR_TEXT_TOKEN, 1)
        detected = _normalize_language_prefix(prefix)
        return {"language": detected or language, "transcription": transcription.strip()}


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _normalize_language_prefix(prefix: str) -> str | None:
    """把 `language English` / `language <English>` / `language none` 归一化成全名。

    没有 `language` 前缀时返回 None（避免把正文首行误判成语种）。
    """
    first_line = next((line for line in prefix.splitlines() if line.strip()), "").strip()
    if not first_line.lower().startswith("language"):
        return None
    rest = first_line[len("language") :].strip()
    if rest.startswith("<") and rest.endswith(">"):
        rest = rest[1:-1].strip()
    if not rest or rest.lower() == "none":
        return None
    return rest


def _load_chat_template(model_path: str, tokenizer) -> str:
    """按 chat_template.json → tokenizer_config.json → tokenizer 属性的顺序取模板。"""
    path = os.path.join(model_path, "chat_template.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        template = payload.get("chat_template")
        if isinstance(template, str) and template:
            return template
        if isinstance(template, list) and template:
            return template[0]
    path = os.path.join(model_path, "tokenizer_config.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        template = payload.get("chat_template")
        if isinstance(template, str) and template:
            return template
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str) and template:
        return template
    raise ValueError(f"在 {model_path} 里找不到 chat template")
