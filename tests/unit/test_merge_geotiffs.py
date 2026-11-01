from pathlib import Path

from pydsm import cmd


def test_merge_geotiffs_preserves_resolution_when_unspecified(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / 'a.tif').touch()
    (tmp_path / 'b.tiff').touch()
    opened = [object(), object()]
    resized: list[float] = []
    merged = object()
    saved: list[object] = []

    monkeypatch.setattr(cmd.geo, 'open_geotiff', lambda _: opened.pop(0))
    monkeypatch.setattr(cmd.geo, 'merge_tiles', lambda tiles: merged)
    monkeypatch.setattr(cmd.geo, 'save_geotiff', lambda dataset, path: saved.append(dataset))

    cmd.merge_geotiffs(str(tmp_path))

    assert resized == []
    assert saved == [merged]


def test_merge_geotiffs_rescales_tiles_before_merging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / 'tile.tif').touch()
    dataset = object()
    calls: list[float] = []

    monkeypatch.setattr(cmd.geo, 'open_geotiff', lambda _: dataset)
    monkeypatch.setattr(
        cmd.geo,
        'rescale',
        lambda value, resolution: calls.append(resolution) or value,
    )
    monkeypatch.setattr(cmd.geo, 'merge_tiles', lambda tiles: dataset)
    monkeypatch.setattr(cmd.geo, 'save_geotiff', lambda *_: None)

    cmd.merge_geotiffs(str(tmp_path), spacial_resolution=0.3)

    assert calls == [0.3]
