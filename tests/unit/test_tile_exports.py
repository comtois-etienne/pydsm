from pathlib import Path

import numpy as np

from pydsm import cmd, const


def test_export_png_tile_writes_four_uint8_channels(tmp_path: Path, monkeypatch) -> None:
    orthophoto = np.array([[[0, 10, 20], [30, 40, 50]]], dtype=np.uint16)
    ndsm = np.array([[0.0, const.CLIP_HEIGHT]], dtype=np.float32)
    datasets = iter((orthophoto, ndsm))
    written: list[tuple[str, np.ndarray]] = []
    monkeypatch.setattr(cmd.tile_cmd.geo, 'open_geotiff', lambda _: object())
    monkeypatch.setattr(cmd.tile_cmd.geo, 'to_ndarray', lambda _: next(datasets))
    monkeypatch.setattr(cmd.tile_cmd.iio, 'imwrite', lambda path, array: written.append((path, array)))

    output = tmp_path / 'rgba' / 'tile.png'
    cmd.export_png_tile('orthophoto.tif', 'ndsm.tif', str(output))

    assert written[0][0] == str(output)
    assert written[0][1].dtype == np.uint8
    assert written[0][1].shape == (1, 2, 4)
    assert written[0][1][0, 1, 3] == 255


def test_export_npz_tiles_uses_expected_directory(tmp_path: Path, monkeypatch) -> None:
    orthophoto_dir = tmp_path / const.ORTHOPHOTO_SUBDIR
    ndsm_dir = tmp_path / const.NDSM_SUBDIR
    orthophoto_dir.mkdir()
    ndsm_dir.mkdir()
    (orthophoto_dir / 'tile.tif').touch()
    (ndsm_dir / 'tile.tif').touch()
    calls: list[str] = []

    def export(*args) -> None:
        calls.append(args[2])

    monkeypatch.setattr(cmd.tile_cmd, 'export_npz_tile', export)
    paths = cmd.export_npz_tiles(str(tmp_path))

    expected = tmp_path / const.RGBA_NPZ_TILES_SUBDIR / 'tile.npz'
    assert paths == [str(expected)]
    assert calls == [str(expected)]


def test_export_png_tile_verbose_reports_output(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    arrays = iter((
        np.zeros((1, 1, 3), dtype=np.uint8),
        np.zeros((1, 1), dtype=np.float32),
    ))
    monkeypatch.setattr(cmd.tile_cmd.geo, 'open_geotiff', lambda _: object())
    monkeypatch.setattr(cmd.tile_cmd.geo, 'to_ndarray', lambda _: next(arrays))
    monkeypatch.setattr(cmd.tile_cmd.iio, 'imwrite', lambda *_: None)
    output = tmp_path / 'tile.png'

    cmd.export_png_tile('orthophoto.tif', 'ndsm.tif', str(output), verbose=True)

    assert str(output) in capsys.readouterr().out
