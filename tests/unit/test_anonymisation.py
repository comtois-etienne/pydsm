import sys
from types import ModuleType

import numpy as np

from pydsm import yolo


class _Boxes:
    xyxy = type(
        '_Coordinates',
        (),
        {'cpu': lambda self: self, 'numpy': lambda self: np.array([[0, 0, 2, 2]])},
    )()


class _Model:
    def predict(self, array, **kwargs):
        assert kwargs['classes'] == [0]
        assert kwargs['conf'] == 0.2
        return [type('_Result', (), {'boxes': _Boxes()})()]


def test_anonymise_uses_direct_ultralytics_prediction(monkeypatch) -> None:
    module = ModuleType('ultralytics')
    module.YOLO = lambda _: _Model()
    monkeypatch.setitem(sys.modules, 'ultralytics', module)

    image = np.arange(12, dtype=np.uint8).reshape((2, 2, 3))
    result = yolo.anonymise_with_yolo(image, 'model.pt')

    assert result.shape == image.shape
    assert result.dtype == np.float64
