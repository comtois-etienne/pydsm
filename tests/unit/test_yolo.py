import numpy as np

from pydsm import yolo


def test_predict_tile_splits_and_recombines_80_meter_tile(monkeypatch) -> None:
    image = np.zeros((8, 8, 4), dtype=np.uint8)
    predictions: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def predict(
        model_paths,
        part: np.ndarray,
        pred_mode: str,
        confidence: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        value = len(predictions) + 1
        instance = np.full(part.shape[:2], value, dtype=np.uint16)
        semantic = np.full(part.shape[:2], value, dtype=np.uint16)
        confidence_map = np.full(part.shape[:2], value, dtype=np.float32)
        result = instance, semantic, confidence_map
        predictions.append(result)
        return result

    monkeypatch.setattr(yolo, 'load_rgbd', lambda *args, **kwargs: image)
    monkeypatch.setattr(yolo.geo, 'open_geotiff', lambda _: object())
    monkeypatch.setattr(yolo.geo, 'get_size', lambda _: (80.0, 80.0))
    monkeypatch.setattr(yolo, 'load_models', lambda models: models)
    monkeypatch.setattr(yolo, 'predict_from_array', predict)
    monkeypatch.setattr(
        yolo.nda,
        'combine_four_instances',
        lambda arrays: np.block([[arrays[0], arrays[1]], [arrays[2], arrays[3]]]),
    )

    result = yolo.predict_tile('model.pt', 'orthophoto.tif', 'ndsm.tif')

    assert len(predictions) == 4
    assert result.instance_labels.shape == (8, 8)
    np.testing.assert_array_equal(result.instance_labels[:4, :4], 1)
    np.testing.assert_array_equal(result.instance_labels[:4, 4:], 2)
    np.testing.assert_array_equal(result.instance_labels[4:, :4], 3)
    np.testing.assert_array_equal(result.instance_labels[4:, 4:], 4)


def test_load_models_injects_loaded_models(monkeypatch) -> None:
    loaded: list[str] = []
    monkeypatch.setattr(yolo, 'load_yolo_model', lambda path: loaded.append(path) or path)

    models = yolo.load_models(['first.pt', 'second.pt'])

    assert models == ['first.pt', 'second.pt']
    assert loaded == ['first.pt', 'second.pt']
