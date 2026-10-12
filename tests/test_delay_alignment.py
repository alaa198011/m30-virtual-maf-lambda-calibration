import numpy as np
import pandas as pd

from fuel_lambda_path import DelayLine, condition_series_like_pseudo_reference
from online_calibration import align_phi_to_pseudo_reference


def test_delay_line_has_exact_requested_sample_delay():
    delay = DelayLine(delay_s=0.02, dt_s=0.02, initial_value=0.0)
    assert [delay.push_pop(v) for v in (1.0, 2.0, 3.0)] == [0.0, 1.0, 2.0]


def test_zero_delay_returns_current_sample():
    delay = DelayLine(delay_s=0.0, dt_s=0.02, initial_value=0.0)
    assert [delay.push_pop(v) for v in (1.0, 2.0)] == [1.0, 2.0]


def test_estimator_alignment_uses_dataset_delay_and_filter_metadata():
    phi = np.array([10.0, 20.0, 30.0, 40.0])
    df = pd.DataFrame({"pseudo_fixed_delay_samples": [1] * 4,
                       "pseudo_filter_alpha": [0.5] * 4})
    expected = condition_series_like_pseudo_reference(phi, 1, 0.5)
    assert np.allclose(align_phi_to_pseudo_reference(df, phi), expected)

