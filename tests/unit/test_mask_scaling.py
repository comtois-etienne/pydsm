import numpy as np

from pydsm import const, nda


def test_scaled_mask_parameters_use_reference_tile() -> None:
    shape = (133, 133)

    assert nda._scaled_mask_parameter(
        const.PIXEL_TOLERANCE,
        shape,
    ) == 2
    assert nda._scaled_mask_parameter(
        const.MIN_MASK_SIZE,
        shape,
        area=True,
    ) == 11
    assert nda._scaled_mask_parameter(
        const.REMOVE_CRACKS_SIZE,
        shape,
    ) == 1


def test_remove_small_masks_scales_area_threshold(monkeypatch) -> None:
    instances = np.zeros((133, 133), dtype=np.uint16)
    instances[0:3, 0:3] = 1
    instances[10:14, 10:14] = 2

    result = nda.remove_small_masks(instances)

    assert result[1, 1] == 0
    assert result[11, 11] == 2
