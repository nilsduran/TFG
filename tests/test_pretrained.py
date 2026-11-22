import numpy as np
import pytest

from br_analysis.pretrained import standard_channel_info, trim_to_patch_size


def test_standard_channel_info_reconstructs_montage_positions() -> None:
    chs_info = standard_channel_info(["Fp1", "Cz", "Oz"], sfreq=200.0)

    assert [channel["ch_name"] for channel in chs_info] == ["Fp1", "Cz", "Oz"]
    assert all(np.isfinite(channel["loc"][:3]).all() for channel in chs_info)


def test_trim_to_patch_size_keeps_complete_patches() -> None:
    X = np.zeros((2, 3, 201), dtype=np.float32)

    trimmed = trim_to_patch_size(X, 200)

    assert trimmed.shape == (2, 3, 200)
    with pytest.raises(ValueError, match="fewer than one"):
        trim_to_patch_size(X[..., :199], 200)
