from pathlib import Path

import numpy as np

from pydsm import cmd


def test_compute_ndsm_dir_uses_tile_subdirectories(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tiles_dir = tmp_path
    dsm_dir = tiles_dir / 'dsm'
    dtm_dir = tiles_dir / 'dtm'
    dsm_dir.mkdir()
    dtm_dir.mkdir()
    (dsm_dir / 'tile.tif').touch()
    (dtm_dir / 'tile.tif').touch()
    calls: list[tuple[str, str, str | None]] = []

    def compute(
        dsm_path: str,
        dtm_path: str,
        ndsm_path: str,
        correct_dtm: bool,
        capture_height: float,
    ) -> str:
        calls.append((dsm_path, dtm_path, ndsm_path))
        return ndsm_path

    monkeypatch.setattr(cmd, 'compute_ndsm', compute)
    monkeypatch.setattr(cmd.geo, 'open_geotiff', lambda _: object())
    monkeypatch.setattr(cmd.geo, 'to_cmap', lambda _: np.zeros((2, 2, 3), dtype=np.uint8))
    written: list[str] = []
    monkeypatch.setattr(cmd.iio, 'imwrite', lambda path, _: written.append(path))

    cmd.compute_ndsm_dir(str(tiles_dir), cmap_dir='cmap')

    assert calls == [(
        str(dsm_dir / 'tile.tif'),
        str(dtm_dir / 'tile.tif'),
        str(tiles_dir / 'ndsm' / 'tile.tif'),
    )]
    assert written == [str(tiles_dir / 'cmap' / 'tile.png')]
