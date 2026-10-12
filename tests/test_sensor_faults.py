import numpy as np

from sensor_models import SensorChannel


def test_detected_dropout_holds_last_valid_sample():
    cfg = {"range": [0, 100], "noise_std": 0.0, "quantization_step": 0.0,
           "dropout_prob": 0.0, "stuck_prob": 0.0}
    channel = SensorChannel(cfg, 0.02, np.random.default_rng(0), "test")
    first, dropped, _ = channel.sample(10.0)
    assert not dropped and first == 10.0
    channel.dropout_rate_per_s = 1e9
    second, dropped, _ = channel.sample(50.0)
    assert dropped and second == first


def test_stuck_status_is_exposed():
    cfg = {"range": [0, 100], "noise_std": 0.0, "quantization_step": 0.0,
           "dropout_rate_per_s": 0.0, "stuck_rate_per_s": 1e9}
    channel = SensorChannel(cfg, 0.02, np.random.default_rng(0), "test")
    _, _, stuck = channel.sample(10.0)
    assert stuck
