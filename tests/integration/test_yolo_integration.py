import os
from pathlib import Path

import pytest

from pydsm.yolo import predict_tile


@pytest.mark.skipif(
    os.getenv('PYDSM_RUN_YOLO_MODEL_TESTS') != '1',
    reason='YOLO model inference is opt-in because it is resource-intensive',
)
def test_predict_tile_processes_80_meter_fixture() -> None:
    files = Path(__file__).parents[2] / 'tests' / 'test_files'
    tile_name = '80_298400_5038480 (2026-11-02).tif'
    model = files / 'models' / 'yolo11n_instance_rgbd_256_f1_60e.pt'

    result = predict_tile(
        str(model),
        str(files / 'tiles' / 'orthophoto' / tile_name),
        str(files / 'tiles' / 'ndsm' / tile_name),
    )

    assert result.instance_labels.shape == (4000, 4000)
    assert result.semantic_labels.shape == (4000, 4000)
    assert result.confidence_labels.shape == (4000, 4000)
