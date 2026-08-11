# ASR 识别不全优化（VAD 调参 + 切块补丁 + 兜底 + 字幕重切）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复"语音稍微小声就识别不全"的问题——通过 VAD 阈值调优、切块 padding、低阈值重扫兜底、响度归一化和字幕长度重切，在不引入长字幕、时间轴错位和内存风险的前提下提升识别完整度。

**Architecture:** 问题链路是 `extract_audio → Silero VAD 分块 → ASR → 断句`。改动分布在四层：①提取层加 loudnorm 响度归一化（把小声对白拉到正常电平）；②VAD 层把 threshold/pad/min_speech 从硬编码默认值改为可配置并调优，切块时带 ±0.3s padding 保留词首词尾；③兜底层：VAD 零检出时用更低阈值二次扫描，而非静默放弃或整段直送 ASR；④输出层加 `reflow_long_segments` 把超长字幕段按标点重切、时间按字符比例分配。所有新参数经 `config.py → task_manager → run_asr → transcribe_with_vad` 传递，并暴露在配置 API 中。已完成的 `/api/asr/diagnostics` 用于事后验证。

**Tech Stack:** Python 3.12 / FastAPI / Silero VAD (silero-vad) / FunASR SenseVoice / Qwen3-ASR / ffmpeg / pytest / pydantic v2

## Global Constraints

- 引擎输入采样率保持 16kHz 单声道（`qwen_asr.SAMPLE_RATE=16000`；降低采样率会丢失擦音高频信息，明确不做）
- 不做"VAD 零检出时整段直送 ASR"（长音频会跑数小时；改用低阈值重扫）
- 所有新配置项加入 `AppConfig` 且带默认值，保证旧 `config.json` 向后兼容（pydantic default）
- 新配置项必须同步加入 `main.py` 的 `ConfigResponse`，否则保存配置时会被丢弃
- 每个改动必须有 pytest 测试，TDD 流程：先写失败测试 → 实现 → 验证通过 → 提交
- 提交粒度：每个 Task 一个 commit，message 遵循仓库现有风格（`feat:` / `fix:` / `test:`）

## File Structure

| 文件 | 职责 | 动作 |
|---|---|---|
| `config.py` | 新增 VAD/归一化/字幕参数 | Modify |
| `core/vad.py` | `detect_speech_segments` 支持 threshold/pad/min_speech | Modify |
| `core/asr/vad_wrapper.py` | 切块 padding、低阈值重扫兜底、`_fix_timestamps` 按比例分配、调用 reflow | Modify |
| `core/asr/base.py` | 新增 `reflow_long_segments`（按句重切 + 时间按字符比例分配） | Modify |
| `core/asr/__init__.py` | `run_asr` 透传 VAD 参数；导出 `reflow_long_segments` | Modify |
| `core/audio.py` | `extract_audio` 支持 loudnorm 归一化 | Modify |
| `core/task_manager.py` | 把 `cfg.vad_*` / `cfg.audio_normalize` 传入 ASR/提取 | Modify |
| `main.py` | `ConfigResponse` 增加新字段 | Modify |
| `tests/test_vad.py` | 新建：VAD 参数转发、padding 切块、重扫兜底、reflow、配置接线 | Create |
| `docs/superpowers/plans/2026-08-11-asr-vad-optimization.md` | 本计划 | Create |

---

### Task 1: VAD 参数可配置 + 默认调优（config + core/vad.py）

**Files:**
- Modify: `config.py`（AppConfig 增加 5 个字段）
- Modify: `core/vad.py:93-114`（`detect_speech_segments` 增加参数并转发）
- Modify: `core/vad.py:141-188`（`split_audio_by_vad` 透传可选参数）
- Test: `tests/test_vad.py`（新建）

**Interfaces:**
- Consumes: 现有 `get_speech_timestamps(audio, model, sampling_rate, return_seconds, min_silence_duration_ms)`
- Produces: `AppConfig.vad_threshold: float = 0.3`、`AppConfig.vad_speech_pad_ms: int = 300`、`AppConfig.vad_min_silence_ms: int = 500`、`AppConfig.vad_min_speech_ms: int = 100`、`AppConfig.max_subtitle_sec: float = 7.0`（Task 4 用）；`detect_speech_segments(audio_path, min_silence_ms=500, threshold=0.3, speech_pad_ms=300, min_speech_ms=100) -> list[SpeechSegment]`

- [ ] **Step 1: 写失败测试（config 默认值 + detect_speech_segments 参数转发）**

`tests/test_vad.py`（新建，含全局 fixtures）：

```python
"""VAD 优化测试。"""

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_vad_model():
    """每测清理模块级模型缓存，避免跨测状态。"""
    import core.vad as vad
    vad._silero_model = None
    yield
    vad._silero_model = None


def test_config_vad_defaults():
    from config import AppConfig
    cfg = AppConfig()
    assert cfg.vad_threshold == 0.3
    assert cfg.vad_speech_pad_ms == 300
    assert cfg.vad_min_silence_ms == 500
    assert cfg.vad_min_speech_ms == 100
    assert cfg.audio_normalize is True
    assert cfg.max_subtitle_sec == 7.0


def test_detect_speech_segments_forwards_tuned_params(tmp_path):
    from core.vad import detect_speech_segments

    wav_path = tmp_path / "t.wav"
    # 用 scipy 写一个 1 秒 16k 静音 WAV，避免依赖 ffmpeg
    import numpy as np
    import scipy.io.wavfile as wavfile
    wavfile.write(str(wav_path), 16000, (np.zeros(16000) * 32767).astype(np.int16))

    fake_ts = [{"start": 0.2, "end": 0.9}]
    with patch("core.vad.load_vad_model") as mock_load, \
         patch("core.vad.get_speech_timestamps", return_value=fake_ts) as mock_ts:
        mock_load.return_value = MagicMock()
        segs = detect_speech_segments(str(wav_path), min_silence_ms=400,
                                      threshold=0.25, speech_pad_ms=500,
                                      min_speech_ms=80)
    assert len(segs) == 1
    assert segs[0].start == 0.2 and segs[0].end == 0.9
    _, kwargs = mock_ts.call_args
    assert kwargs["threshold"] == 0.25
    assert kwargs["speech_pad_ms"] == 500
    assert kwargs["min_speech_duration_ms"] == 80
    assert kwargs["min_silence_duration_ms"] == 400
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_vad.py -v`
Expected: FAIL（`detect_speech_segments` 不接受 `threshold` 参数，TypeError；`AppConfig` 无 `vad_threshold`）

- [ ] **Step 3: 实现**

`config.py` AppConfig 增加（放在 `asr_language` 之后、`asr_api_url` 之前）：

```python
    asr_model_online: str = ""          # 在线模型名

    # VAD 配置（用于修复"小声语音识别不全"）
    vad_threshold: float = 0.3           # Silero 语音概率阈值，越低越敏感（默认 0.5 会漏低 SNR 语音）
    vad_speech_pad_ms: int = 300         # 语音段前后 padding(ms)，保住词首词尾（默认 30ms 太小）
    vad_min_silence_ms: int = 500        # 最小静音(ms)，用于切分
    vad_min_speech_ms: int = 100         # 最小语音(ms)，过滤瞬态噪声
    max_subtitle_sec: float = 7.0        # 单条字幕最大时长(秒)，超限按句重切（Task 4 消费）
    audio_normalize: bool = True         # 提取音频时 loudnorm 响度归一化（Task 5 消费）
```

`core/vad.py` `detect_speech_segments` 改为：

```python
def detect_speech_segments(
    audio_path: str,
    min_silence_ms: int = 500,
    threshold: float = 0.3,
    speech_pad_ms: int = 300,
    min_speech_ms: int = 100,
) -> list[SpeechSegment]:
    """
    使用 Silero VAD 检测音频中的语音片段。

    音频已是提取后的 WAV 文件（非原始视频），直接一次性读取即可。
    threshold 越低越能检出小声/低 SNR 语音；speech_pad_ms 越大越能保住词首词尾。
    """
    from silero_vad import get_speech_timestamps

    model = load_vad_model()
    wav, sr = _read_audio(audio_path)
    if wav.numel() == 0:
        return []

    timestamps = get_speech_timestamps(
        wav,
        _silero_model,
        sampling_rate=sr,
        return_seconds=True,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
    )
    segments = [SpeechSegment(start=ts["start"], end=ts["end"]) for ts in timestamps]
    total = sum(s.end - s.start for s in segments)
    logger.info("VAD detected %d speech segments (%.1fs total) in %s (threshold=%.2f, pad=%dms)",
                len(segments), total, os.path.basename(audio_path), threshold, speech_pad_ms)
    return segments
```

`split_audio_by_vad` 签名保持默认值兼容（仅透传，Task 2 会重建切块逻辑，此函数可暂不改）：

```python
def split_audio_by_vad(
    audio_path: str,
    output_dir: str,
    min_silence_ms: int = 500,
    min_segment_sec: float = 0.5,
    threshold: float = 0.3,
    speech_pad_ms: int = 300,
    min_speech_ms: int = 100,
) -> list[tuple[str, float, float]]:
    ...
    segments = detect_speech_segments(audio_path, min_silence_ms, threshold, speech_pad_ms, min_speech_ms)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_vad.py -v`
Expected: PASS（2 个测试）

- [ ] **Step 5: 提交**

```bash
git add config.py core/vad.py tests/test_vad.py
git commit -m "feat: tune VAD params (threshold/pad/min_speech) with configurable defaults"
```

---

### Task 2: VAD 切块加 padding，保住词首词尾

**Files:**
- Modify: `core/asr/vad_wrapper.py:18-150`（`transcribe_with_vad` 的切块循环）
- Test: `tests/test_vad.py`（追加）

**Interfaces:**
- Consumes: `detect_speech_segments(audio_path, min_silence_ms, threshold, speech_pad_ms, min_speech_ms)`（Task 1）
- Produces: `transcribe_with_vad(engine, audio_path, min_silence_ms=500, threshold=0.3, speech_pad_ms=300, min_speech_ms=100, language="auto", pad_sec=0.3) -> tuple[list[dict], str]`。行为：每个 chunk 用 `-ss max(0, seg.start-pad_sec) -to min(dur, seg.end+pad_sec)` 提取，chunk 内时间戳偏移量用 **cut_start**（而非 seg.start）。

- [ ] **Step 1: 写失败测试（切块命令含 padding，且时间戳按 cut_start 偏移）**

追加到 `tests/test_vad.py`：

```python
def test_transcribe_with_vad_chunks_use_padding(tmp_path):
    """切块提取应带 ±pad_sec padding，时间戳按实际切点偏移。"""
    import core.asr.vad_wrapper as vw
    from core.asr.base import AsrEngine

    fake_engine = AsrEngine()
    fake_engine.transcribe = MagicMock(return_value=(
        [{"start": 0.0, "end": 1.0, "text": "hello"}], "en"))

    wav_path = tmp_path / "t.wav"
    import numpy as np
    import scipy.io.wavfile as wavfile
    wavfile.write(str(wav_path), 16000, (np.zeros(16000 * 5) * 32767).astype(np.int16))

    segs = [
        type("S", (), {"start": 2.0, "end": 3.5})(),
    ]
    with patch("core.asr.vad_wrapper.detect_speech_segments", return_value=segs), \
         patch("core.asr.vad_wrapper.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        segments, _ = vw.transcribe_with_vad(
            fake_engine, str(wav_path), min_silence_ms=500,
            threshold=0.3, speech_pad_ms=300, min_speech_ms=100, pad_sec=0.3,
        )
    # 命令应使用 padding 后的切点
    cmd = mock_run.call_args.args[0]
    assert "-ss" in cmd and "1.7" in cmd  # 2.0 - 0.3
    assert "-to" in cmd and "3.8" in cmd  # 3.5 + 0.3
    # 时间戳按 cut_start=1.7 偏移
    assert segments[0]["start"] == round(0.0 + 1.7, 3)
    assert segments[0]["end"] == round(1.0 + 1.7, 3)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_vad.py::test_transcribe_with_vad_chunks_use_padding -v`
Expected: FAIL（当前切块用 `seg.start` 无 padding，`-ss` 后是 `2.0` 不是 `1.7`）

- [ ] **Step 3: 实现**

`core/asr/vad_wrapper.py` 修改：

```python
def transcribe_with_vad(
    engine: AsrEngine,
    audio_path: str,
    min_silence_ms: int = 500,
    threshold: float = 0.3,
    speech_pad_ms: int = 300,
    min_speech_ms: int = 100,
    language: str = "auto",
    pad_sec: float = 0.3,
) -> tuple[list[dict], str]:
    """
    使用 VAD 分块处理长音频并识别。

    pad_sec: 切块时每侧额外包含的秒数，避免硬切丢失词首/词尾（VAD 起检会滞后）。
    """
    from core.vad import detect_speech_segments
    from core.audio import get_audio_duration

    speech_segments = detect_speech_segments(
        audio_path, min_silence_ms, threshold, speech_pad_ms, min_speech_ms
    )

    if not speech_segments:
        # Task 3 会在此处加低阈值重扫
        logger.info("VAD found no speech in %s, skipping ASR",
                     os.path.basename(audio_path))
        return [], ""

    duration = get_audio_duration(audio_path) or (
        speech_segments[-1].end + pad_sec
    )

    total_speech = sum(s.end - s.start for s in speech_segments)

    if len(speech_segments) == 1 or total_speech < 30:
        logger.info("VAD: total speech %.1fs < 30s, processing as single chunk (no split)", total_speech)
        segments, detected_lang = engine.transcribe(audio_path, language=language)
        return _fix_timestamps(segments, speech_segments, audio_path), detected_lang
    ...
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, seg in enumerate(speech_segments):
            # 带 padding 的切点，并夹到音频范围
            cut_start = max(0.0, seg.start - pad_sec)
            cut_end = min(duration, seg.end + pad_sec)
            chunk_path = os.path.join(tmp_dir, f"{prefix}_chunk_{i:04d}.wav")
            import subprocess
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-ss", str(round(cut_start, 3)),
                "-to", str(round(cut_end, 3)),
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                "-loglevel", "error",
                chunk_path,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode != 0:
                logger.warning("Failed to extract chunk %d", i)
                continue
            ...
            # 时间戳还原：偏移基准是 cut_start（含 padding 的起点），不是 seg.start
            if len(chunk_segments) == 1 and chunk_segments[0]["start"] == 0.0:
                chunk_segments[0]["start"] = round(cut_start, 3)
                chunk_segments[0]["end"] = round(cut_end, 3)
            for s in chunk_segments:
                s["start"] = round(s["start"] + cut_start, 3)
                s["end"] = round(s["end"] + cut_start, 3)
                all_segments.append(s)
            logger.info("VAD chunk %d/%d: %.1f-%.1fs -> cut %.1f-%.1fs → %d segments (lang=%s)",
                        i + 1, len(speech_segments),
                        seg.start, seg.end, cut_start, cut_end,
                        len(chunk_segments), chunk_lang)
            ...
```

注意：`segs = [type("S", (), {"start": 2.0, "end": 3.5})()]` 在测试里替代 `SpeechSegment`（dataclass 不需要构造参数），实现里 `seg.start` / `seg.end` 属性访问不变。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_vad.py -v`
Expected: PASS（3 个测试）

- [ ] **Step 5: 提交**

```bash
git add core/asr/vad_wrapper.py tests/test_vad.py
git commit -m "feat: pad VAD chunk cuts by 0.3s to preserve word edges"
```

---

### Task 3: VAD 零检出时低阈值重扫兜底

**Files:**
- Modify: `core/asr/vad_wrapper.py:30-36`（零检出分支）
- Test: `tests/test_vad.py`（追加）

**Interfaces:**
- Consumes: `detect_speech_segments`（Task 1 已支持低阈值参数）
- Produces: 行为——第一次 `detect_speech_segments` 返回空时，用 `threshold=max(threshold*0.6, 0.1)`、`min_speech_ms=min(min_speech_ms, 50)` 重扫一次；仍为空才返回 `[], ""`。重扫命中时打 INFO 日志。

- [ ] **Step 1: 写失败测试**

```python
def test_transcribe_with_vad_rescans_with_lower_threshold(tmp_path):
    """VAD 零检出时应以更低阈值重扫，而不是直接放弃。"""
    import core.asr.vad_wrapper as vw
    from core.asr.base import AsrEngine

    fake_engine = AsrEngine()
    fake_engine.transcribe = MagicMock(return_value=(
        [{"start": 0.0, "end": 1.0, "text": "hi"}], "en"))

    import numpy as np
    import scipy.io.wavfile as wavfile
    wav_path = tmp_path / "t.wav"
    wavfile.write(str(wav_path), 16000, (np.zeros(16000) * 32767).astype(np.int16))

    first_call = {"args": None}

    def fake_detect(audio_path, min_silence_ms=500, threshold=0.3,
                    speech_pad_ms=300, min_speech_ms=100):
        if first_call["args"] is None:
            first_call["args"] = (threshold, min_speech_ms)
            return []
        return [type("S", (), {"start": 0.0, "end": 1.0})()]

    with patch("core.asr.vad_wrapper.detect_speech_segments", side_effect=fake_detect), \
         patch("core.asr.vad_wrapper.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        segments, _ = vw.transcribe_with_vad(
            fake_engine, str(wav_path), threshold=0.3,
            speech_pad_ms=300, min_speech_ms=100,
        )
    # 第一次用了原始阈值；第二次应更低
    assert first_call["args"][0] == 0.3
    assert len(segments) == 1


def test_transcribe_with_vad_rescan_still_empty_returns_empty(tmp_path):
    """重扫仍为空时应返回空列表（不抛异常、不整段直送）。"""
    import core.asr.vad_wrapper as vw
    from core.asr.base import AsrEngine

    fake_engine = AsrEngine()
    import numpy as np
    import scipy.io.wavfile as wavfile
    wav_path = tmp_path / "t.wav"
    wavfile.write(str(wav_path), 16000, (np.zeros(16000) * 32767).astype(np.int16))

    with patch("core.asr.vad_wrapper.detect_speech_segments", return_value=[]):
        segments, _ = vw.transcribe_with_vad(fake_engine, str(wav_path))
    assert segments == []
    fake_engine.transcribe.assert_not_called()  # 不整段直送
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_vad.py::test_transcribe_with_vad_rescans_with_lower_threshold tests/test_vad.py::test_transcribe_with_vad_rescan_still_empty_returns_empty -v`
Expected: FAIL（当前零检出直接 `return [], ""`，不重扫）

- [ ] **Step 3: 实现**

`core/asr/vad_wrapper.py` 零检出分支改为：

```python
    speech_segments = detect_speech_segments(
        audio_path, min_silence_ms, threshold, speech_pad_ms, min_speech_ms
    )

    if not speech_segments:
        # 兜底：默认阈值检不到时用更低阈值重扫，专门捞小声/低 SNR 语音。
        # 不做"整段直送 ASR"——长音频会跑数小时。
        rescan_threshold = max(threshold * 0.6, 0.1)
        rescan_min_speech = min(min_speech_ms, 50)
        logger.info(
            "VAD found no speech at threshold=%.2f, rescanning with threshold=%.2f",
            threshold, rescan_threshold,
        )
        speech_segments = detect_speech_segments(
            audio_path, min_silence_ms,
            rescan_threshold, speech_pad_ms, rescan_min_speech,
        )
        if speech_segments:
            total = sum(s.end - s.start for s in speech_segments)
            logger.info("VAD rescan recovered %d segments (%.1fs) at lower threshold",
                        len(speech_segments), total)
        else:
            logger.info("VAD found no speech in %s (even at threshold=%.2f), skipping ASR",
                        os.path.basename(audio_path), rescan_threshold)
            return [], ""
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_vad.py -v`
Expected: PASS（5 个测试）

- [ ] **Step 5: 提交**

```bash
git add core/asr/vad_wrapper.py tests/test_vad.py
git commit -m "feat: rescan VAD with lower threshold when no speech found"
```

---

### Task 4: 字幕长度重切 `reflow_long_segments` + 修复 `_fix_timestamps` 分配

**Files:**
- Modify: `core/asr/base.py`（新增 `reflow_long_segments`，放在 `add_silence_gaps` 之后）
- Modify: `core/asr/__init__.py`（导出 `reflow_long_segments`）
- Modify: `core/asr/vad_wrapper.py`（长音频路径在 `add_silence_gaps` 前调用 reflow；`_fix_timestamps` 改为按 VAD 时长比例分配句子并 reflow）
- Test: `tests/test_vad.py`（追加）

**Interfaces:**
- Consumes: `_SENTENCE_PUNCT`（base.py 已定义）、`cfg.max_subtitle_sec`（Task 1）
- Produces: `reflow_long_segments(segments: list[dict], max_sec: float = 7.0, max_chars: int = 60) -> list[dict]`——对超过 `max_sec` 或 `max_chars` 的段，按标点拆句、时间按字符数比例分配；拆不动（无标点）或本就不超限的段原样保留。

- [ ] **Step 1: 写失败测试**

```python
def test_reflow_long_segment_splits_by_punctuation():
    from core.asr.base import reflow_long_segments
    segs = [{"start": 0.0, "end": 8.0, "text": "第一句。第二句。第三句。"}]
    out = reflow_long_segments(segs, max_sec=5.0, max_chars=100)
    assert len(out) == 3
    assert out[0]["text"] == "第一句。"
    assert out[-1]["end"] == 8.0
    # 时间按字符比例：3 句等长 → 每句 8/3 s
    assert abs(out[1]["start"] - 8.0 / 3) < 0.01
    assert abs(out[1]["end"] - 16.0 / 3) < 0.01


def test_reflow_keeps_short_and_unsplittable_segments():
    from core.asr.base import reflow_long_segments
    segs = [
        {"start": 0.0, "end": 2.0, "text": "短句。"},
        {"start": 2.0, "end": 20.0, "text": "没有标点的超长连续文本没有标点的超长连续文本没有标点"},  # 无标点
    ]
    out = reflow_long_segments(segs, max_sec=5.0, max_chars=30)
    assert out[0] == segs[0]
    assert out[1] == segs[1]  # 拆不动原样保留


def test_reflow_distributes_by_char_proportion():
    from core.asr.base import reflow_long_segments
    # 两句：8 字 + 2 字，共 10 字，总时长 5s → 4s / 1s
    segs = [{"start": 0.0, "end": 5.0, "text": "一二三四五六七八。九零。"}]
    out = reflow_long_segments(segs, max_sec=3.0, max_chars=100)
    assert len(out) == 2
    assert abs(out[0]["end"] - 4.0) < 0.01
    assert abs(out[1]["start"] - 4.0) < 0.01
    assert out[1]["end"] == 5.0


def test_fix_timestamps_distributes_proportionally(tmp_path):
    """_fix_timestamps 应按 VAD 时长比例分配句子，而不是剩余全倒进最后一段。"""
    import core.asr.vad_wrapper as vw
    segments = [{"start": 0.0, "end": 0.0, "text": "第一句。第二句。第三句。第四句。"}]
    speech_segments = [
        type("S", (), {"start": 0.0, "end": 4.0})(),
        type("S", (), {"start": 5.0, "end": 6.0})(),
    ]
    out = vw._fix_timestamps(segments, speech_segments, "/tmp/t.wav")
    # 4 句话分配到 2 段：按 4:1 时长比例 → 第一段 3~4 句，第二段 0~1 句
    assert len(out) == 2
    assert out[0]["text"].startswith("第一句")
    assert out[0]["start"] == 0.0 and out[0]["end"] == 4.0
    assert out[1]["start"] == 5.0 and out[1]["end"] == 6.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_vad.py -v`
Expected: FAIL（`reflow_long_segments` 不存在 → ImportError；`_fix_timestamps` 目前 1-2 句/段 + 剩余倒灌）

- [ ] **Step 3: 实现**

`core/asr/base.py` 新增（放在 `add_silence_gaps` 之后）：

```python
def reflow_long_segments(
    segments: list[dict],
    max_sec: float = 7.0,
    max_chars: int = 60,
) -> list[dict]:
    """把过长的字幕段按标点重切，时间按字符数比例分配。

    用于 ASR 未返回时间戳（或 VAD 合并导致段过长）时的兜底：
    每段内部按字符比例插值时间，保证"单条字幕不长"且总时长不变。
    无标点可拆的段原样保留（避免粗暴均分）。
    """
    out: list[dict] = []
    for seg in segments:
        dur = seg["end"] - seg["start"]
        text = seg.get("text", "").strip()
        if dur <= max_sec and len(text) <= max_chars:
            out.append(seg)
            continue

        parts = [p.strip() for p in _SENTENCE_PUNCT.split(text) if p.strip()]
        if len(parts) < 2:
            out.append(seg)  # 拆不动
            continue

        total_chars = sum(len(p) for p in parts)
        t = seg["start"]
        for part in parts:
            span = dur * len(part) / total_chars
            out.append({
                "start": round(t, 3),
                "end": round(t + span, 3),
                "text": part,
            })
            t += span
    return out
```

`core/asr/__init__.py`：`from core.asr.base import (..., reflow_long_segments)` 并加入 `__all__`。

`core/asr/vad_wrapper.py`：
- 长音频路径末尾（`all_segments.sort(...)` 之后、`add_silence_gaps` 之前）插入：

```python
    # 兜底：超长字幕段按句重切（ASR 未给时间戳时，长 chunk 会变成单条超长字幕）
    from core.asr.base import reflow_long_segments
    all_segments = reflow_long_segments(all_segments)
```

- 重写 `_fix_timestamps` 的分配循环（保持签名 `(segments, speech_segments, audio_path)` 与"无时间戳才处理"的早退逻辑）：

```python
    # 按 VAD 片段时长比例分配句子，避免剩余句子全倒进最后一段
    total_dur = sum(max(0.3, s.end - s.start) for s in speech_segments) or 1.0
    total_chars = sum(len(s) for s in sentences) or 1
    result = []
    sent_idx = 0
    for vad_seg in speech_segments:
        if sent_idx >= len(sentences):
            break
        budget = max(1, int(total_chars * max(0.3, vad_seg.end - vad_seg.start) / total_dur))
        group = []
        while sent_idx < len(sentences) and sum(len(x) for x in group) < budget:
            group.append(sentences[sent_idx])
            sent_idx += 1
        if not group:
            continue
        result.append({
            "start": round(vad_seg.start, 3),
            "end": round(vad_seg.end, 3),
            "text": " ".join(group),
        })

    # 剩余句子并入最后一段（正常不应发生；比例分配已按时长兜底）
    if sent_idx < len(sentences) and result:
        result[-1]["text"] += " " + " ".join(sentences[sent_idx:])

    if not result:
        return segments

    # 重切兜底：分配后仍有超长段则按句再拆
    from core.asr.base import reflow_long_segments
    return reflow_long_segments(result)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_vad.py -v`
Expected: PASS（9 个测试）

- [ ] **Step 5: 提交**

```bash
git add core/asr/base.py core/asr/__init__.py core/asr/vad_wrapper.py tests/test_vad.py
git commit -m "feat: reflow long subtitle segments by punctuation; proportional _fix_timestamps"
```

---

### Task 5: 提取音频响度归一化（loudnorm）

**Files:**
- Modify: `core/audio.py:10-35`（`extract_audio` 支持 `normalize` 参数）
- Modify: `core/task_manager.py`（把 `cfg.audio_normalize` 传入）
- Test: `tests/test_vad.py`（追加）

**Interfaces:**
- Consumes: `AppConfig.audio_normalize: bool = True`（Task 1）
- Produces: `extract_audio(media_path, output_path, normalize: bool = True) -> bool`——`normalize=True` 时在 ffmpeg 命令中插入 `-af loudnorm=I=-23:TP=-1.5:LRA=11`；`task_manager._execute_pipeline` 调用处传 `cfg.audio_normalize`。

- [ ] **Step 1: 写失败测试**

```python
def test_extract_audio_adds_loudnorm_when_normalize(tmp_path):
    import asyncio
    from core.audio import extract_audio

    src = tmp_path / "in.mp4"
    dst = tmp_path / "out.wav"
    src.write_bytes(b"fake")
    with patch("core.audio.subprocess.Popen") as mock_popen:
        proc = MagicMock()
        proc.communicate.return_value = (b"", b"")
        proc.returncode = 0
        mock_popen.return_value = proc
        asyncio.run(extract_audio(str(src), str(dst), normalize=True))
    cmd = mock_popen.call_args.args[0]
    assert "loudnorm=I=-23:TP=-1.5:LRA=11" in cmd
    assert "-af" in cmd


def test_extract_audio_skips_loudnorm_when_disabled(tmp_path):
    import asyncio
    from core.audio import extract_audio

    src = tmp_path / "in.mp4"
    dst = tmp_path / "out.wav"
    src.write_bytes(b"fake")
    with patch("core.audio.subprocess.Popen") as mock_popen:
        proc = MagicMock()
        proc.communicate.return_value = (b"", b"")
        proc.returncode = 0
        mock_popen.return_value = proc
        asyncio.run(extract_audio(str(src), str(dst), normalize=False))
    cmd = mock_popen.call_args.args[0]
    assert "loudnorm" not in cmd
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_vad.py -k extract_audio -v`
Expected: FAIL（`extract_audio` 不接受 `normalize` 参数）

- [ ] **Step 3: 实现**

`core/audio.py`：

```python
async def extract_audio(media_path: str, output_path: str, normalize: bool = True) -> bool:
    """从视频中提取音频，转为 16kHz 单声道 WAV。

    normalize=True 时加 loudnorm 响度归一化：把小声对白拉到与正常对白
    相近的电平，降低 VAD 漏检和 ASR 误识率（不改变采样率，16kHz 保留
    擦音高频信息）。
    """
    cmd = [
        "ffmpeg",
        "-i", media_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        output_path,
    ]
    if normalize:
        # -af 滤镜需放在输出文件之前
        cmd.insert(-2, "-af")
        cmd.insert(-2, "loudnorm=I=-23:TP=-1.5:LRA=11")
    logger.info("Extracting audio: %s -> %s (normalize=%s)", media_path, output_path, normalize)
    ...
```

`core/task_manager.py` 调用处：

```python
            ok = asyncio.new_event_loop().run_until_complete(
                extract_audio(video_path, audio_path, normalize=cfg.audio_normalize)
            )
```

- [ ] **Step 4: 跑测试确认通过 + 真实 ffmpeg 冒烟**

Run: `.venv/bin/python -m pytest tests/test_vad.py -v`
Expected: PASS

再跑真实冒烟（本机已装 ffmpeg）确认 loudnorm 不会破坏输出：

```bash
.venv/bin/python -m pytest tests/test_real_media.py::test_extract_audio_real_mp4 -v
```

Expected: PASS（提取出的 WAV 非空、可解码）

- [ ] **Step 5: 提交**

```bash
git add core/audio.py core/task_manager.py tests/test_vad.py
git commit -m "feat: loudnorm loudness normalization during audio extraction"
```

---

### Task 6: 配置接线（config → task_manager → run_asr）+ ConfigResponse

**Files:**
- Modify: `core/asr/__init__.py`（`run_asr` 增加 VAD 参数并透传）
- Modify: `core/task_manager.py`（`run_asr` 调用处传入 `cfg.vad_*`）
- Modify: `main.py`（`ConfigResponse` 增加新字段，防止保存配置时被丢弃）
- Test: `tests/test_vad.py`（追加）

**Interfaces:**
- Consumes: Task 1-5 的全部参数与函数
- Produces: `run_asr(audio_path, mode="local", model_name=..., asr_language="auto", api_url="", api_key="", model_online="", engine="qwen3-asr", use_vad=False, vad_min_silence_ms=500, vad_threshold=0.3, vad_speech_pad_ms=300, vad_min_speech_ms=100) -> tuple[list[dict], str]`

- [ ] **Step 1: 写失败测试**

```python
def test_run_asr_forwards_vad_params(tmp_path):
    """run_asr 应把 VAD 参数透传给 transcribe_with_vad。"""
    from core.asr import run_asr

    wav_path = tmp_path / "t.wav"
    import numpy as np
    import scipy.io.wavfile as wavfile
    wavfile.write(str(wav_path), 16000, (np.zeros(16000) * 32767).astype(np.int16))

    with patch("core.asr.get_asr_engine") as mock_eng, \
         patch("core.asr.vad_wrapper.transcribe_with_vad", return_value=([], "")) as mock_twv:
        mock_eng.return_value = type("E", (), {"need_vad": lambda self: True})()
        run_asr(str(wav_path), engine="qwen3-asr", use_vad=True,
                vad_min_silence_ms=400, vad_threshold=0.25,
                vad_speech_pad_ms=500, vad_min_speech_ms=80)
    _, kwargs = mock_twv.call_args
    assert kwargs["min_silence_ms"] == 400
    assert kwargs["threshold"] == 0.25
    assert kwargs["speech_pad_ms"] == 500
    assert kwargs["min_speech_ms"] == 80


def test_config_response_includes_vad_fields():
    """ConfigResponse 必须包含新字段，避免保存配置时被 pydantic 丢弃。"""
    from main import ConfigResponse
    body = ConfigResponse.model_validate({
        "asr_mode": "local", "asr_model": "m", "asr_api_url": "",
        "asr_api_key": "", "asr_model_online": "",
        "translate_mode": "local", "translate_api_url": "",
        "translate_api_key": "", "translate_model": "",
        "translate_model_local": "", "translate_prompt_format": "json",
        "translate_thinking": False, "target_language": "zh-CN",
        "path_mappings": {}, "temp_dir": "./tmp", "video_dirs": [],
        "vad_threshold": 0.2, "vad_speech_pad_ms": 400,
        "vad_min_silence_ms": 300, "vad_min_speech_ms": 60,
        "max_subtitle_sec": 6.0, "audio_normalize": False,
    })
    assert body.vad_threshold == 0.2
    assert body.audio_normalize is False
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_vad.py -k "run_asr or config_response" -v`
Expected: FAIL（`run_asr` 不接受新参数；`ConfigResponse` 无 `vad_threshold` 字段）

- [ ] **Step 3: 实现**

`core/asr/__init__.py` `run_asr` 签名与调用：

```python
def run_asr(
    audio_path: str,
    mode: str = "local",
    model_name: str = "Qwen/Qwen3-ASR-0.6B",
    asr_language: str = "auto",
    api_url: str = "",
    api_key: str = "",
    model_online: str = "",
    engine: str = "qwen3-asr",
    use_vad: bool = False,
    vad_min_silence_ms: int = 500,
    vad_threshold: float = 0.3,
    vad_speech_pad_ms: int = 300,
    vad_min_speech_ms: int = 100,
) -> tuple[list[dict], str]:
    ...
    if use_vad and eng.need_vad():
        from core.asr.vad_wrapper import transcribe_with_vad
        return transcribe_with_vad(
            eng, audio_path,
            min_silence_ms=vad_min_silence_ms,
            threshold=vad_threshold,
            speech_pad_ms=vad_speech_pad_ms,
            min_speech_ms=vad_min_speech_ms,
            language=asr_language,
        )
```

`core/task_manager.py` 调用处（`run_asr(...)` 增加）：

```python
                    segments, detected_lang = run_asr(
                        audio_path,
                        mode=cfg.asr_mode,
                        engine=cfg.asr_engine,
                        model_name=cfg.asr_model,
                        asr_language=task["asr_language"] if task.get("asr_language") else cfg.asr_language,
                        api_url=cfg.asr_api_url,
                        api_key=cfg.asr_api_key,
                        model_online=cfg.asr_model_online,
                        use_vad=True,
                        vad_min_silence_ms=cfg.vad_min_silence_ms,
                        vad_threshold=cfg.vad_threshold,
                        vad_speech_pad_ms=cfg.vad_speech_pad_ms,
                        vad_min_speech_ms=cfg.vad_min_speech_ms,
                    )
```

`main.py` `ConfigResponse` 增加字段（与 `config.py` 一致）：

```python
    asr_model_online: str
    vad_threshold: float = 0.3
    vad_speech_pad_ms: int = 300
    vad_min_silence_ms: int = 500
    vad_min_speech_ms: int = 100
    max_subtitle_sec: float = 7.0
    audio_normalize: bool = True
```

同时 `api_save_config` 中 `AppConfig(**body.model_dump())` 自动带上新字段（Task 1 已加默认值，无需改动该函数本身）。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 全绿（除本机已知的 ffmpeg/环境无关项外；本机已装 ffmpeg，应全绿）

- [ ] **Step 5: 提交**

```bash
git add core/asr/__init__.py core/task_manager.py main.py tests/test_vad.py
git commit -m "feat: wire VAD config params through task pipeline and config API"
```

---

### Task 7: 端到端验证 + 文档

**Files:**
- Modify: `README.md`（VAD 调优 + 诊断使用说明）
- Test: `tests/test_vad.py`（如需要补 edge case）

**Interfaces:**
- Consumes: 全部 Task 交付物

- [ ] **Step 1: 全套件回归**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 全部 PASS（含 Task 1-6 新增测试；`tests/test_asr_diagnostics.py` 8 个 + `tests/test_vad.py` 13 个）

- [ ] **Step 2: 真实音频冒烟（可选，验证 VAD 参数效果）**

Run:
```bash
.venv/bin/python -c "
import av, numpy as np, torch
from silero_vad import load_silero_vad, get_speech_timestamps
# 用 en.mp4 验证 threshold=0.3 + pad=300 相比默认 0.5/30 的检出变化
"
```
Expected: 脚本打印两种参数下检出的语音段数/总时长，确认低阈值不减少检出（参考分析期数据：th=0.5 检出 51.7s；低阈值 + 大 pad 在低 SNR 场景恢复 0→N 段）

- [ ] **Step 3: README 更新**

在 `README.md` 增加一节：

```markdown
## VAD 与识别完整度调优

识别不全（尤其小声/低 SNR 对白）通常由 VAD 漏检引起。可通过 `config.json`
调整：

| 参数 | 默认 | 说明 |
|---|---|---|
| `vad_threshold` | 0.3 | Silero 语音概率阈值，越低越敏感（默认 0.5 会漏小声语音） |
| `vad_speech_pad_ms` | 300 | 语音段 padding，保住词首词尾 |
| `vad_min_silence_ms` | 500 | 最小静音，控制切分粒度 |
| `vad_min_speech_ms` | 100 | 最小语音时长，过滤瞬态噪声 |
| `audio_normalize` | true | 提取时 loudnorm 响度归一化 |
| `max_subtitle_sec` | 7.0 | 单条字幕最大时长，超限按句重切 |

诊断：`GET /api/asr/diagnostics` 返回各引擎时间戳统计
（`sensevoice.none` 占比高说明字幕时间轴大量退回估算）。
```

- [ ] **Step 4: 提交**

```bash
git add README.md
git commit -m "docs: document VAD tuning and ASR diagnostics"
```

---

## Self-Review 记录

**1. Spec 覆盖检查**（对照需求"小声语音识别不全"）：
- VAD 漏检 → Task 1（阈值 0.3）+ Task 3（低阈值重扫）+ Task 5（loudnorm 提升 SNR）✓
- 词首词尾被切 → Task 2（切块 padding）✓
- 长 chunk → 单条长字幕 → Task 4（reflow + `_fix_timestamps` 比例分配）✓
- 时间轴错位顾虑 → Task 4 只在 `start==end`（无真实时间戳）路径 reflow，有真实时间戳的段不碰 ✓
- 内存/超时顾虑 → 明确不做整段直送（Task 3 重扫替代）✓
- 降 kHz 有害 → 不做，Task 5 用 loudnorm（保留 16kHz）✓
- 已完成的诊断（`/api/asr/diagnostics`、subtitle_checker 修复、ffmpeg 安装）不在本计划内，直接作为前置条件 ✓

**2. 占位符扫描**：所有 Step 均含具体代码/命令，无 TBD/TODO。✓

**3. 类型/签名一致性**：
- `detect_speech_segments(audio_path, min_silence_ms, threshold, speech_pad_ms, min_speech_ms)` —— Task 1 定义，Task 2/3 消费，参数顺序一致 ✓
- `transcribe_with_vad(engine, audio_path, min_silence_ms, threshold, speech_pad_ms, min_speech_ms, language, pad_sec)` —— Task 2 定义，Task 3 复用，Task 6 的 `run_asr` 按关键字传参 ✓
- `reflow_long_segments(segments, max_sec=7.0, max_chars=60)` —— Task 4 定义并导出，`vad_wrapper` 与 `_fix_timestamps` 消费 ✓
- `extract_audio(media_path, output_path, normalize=True)` —— Task 5 定义，`task_manager` 消费 ✓
- `AppConfig` 新字段名与 `ConfigResponse` 完全一致（vad_threshold/vad_speech_pad_ms/vad_min_silence_ms/vad_min_speech_ms/max_subtitle_sec/audio_normalize）✓
