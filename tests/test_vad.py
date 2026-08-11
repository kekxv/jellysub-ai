"""VAD 优化测试：参数转发、切块 padding、低阈值重扫、字幕重切、配置接线。"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_vad_model():
    """每测清理模块级模型缓存，避免跨测状态。"""
    import core.vad as vad
    vad._silero_model = None
    yield
    vad._silero_model = None


def _write_wav(path, seconds=1.0, rate=16000):
    """写一个静音 WAV，避免依赖 ffmpeg。"""
    import numpy as np
    import scipy.io.wavfile as wavfile
    wavfile.write(str(path), rate, (np.zeros(int(rate * seconds)) * 32767).astype(np.int16))


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
    _write_wav(wav_path)

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
