from datetime import date
from pathlib import Path

from pydsm import cmd


class _City:
    def get_bounding_box(self) -> tuple[float, float, float, float]:
        return 0, 85, 0, 85

    def epsg(self) -> int:
        return 2950

    def is_coordinate_inside(
        self,
        coordinate: list[float],
        bounding_box: bool,
    ) -> bool:
        assert bounding_box is False
        return 0 <= coordinate[0] <= 85 and 0 <= coordinate[1] <= 85


def test_load_tile_in_cityjson_filters_tiles_by_corners(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(cmd.cjio, 'read_cityjson', lambda _: _City())
    downloaded: list[tuple[float, float]] = []

    def download_tile(
        tiles_dir: str,
        name: str,
        epsg: int,
        origin: tuple[float, float],
        size: float,
        spacial_resolution: float,
        verbose: bool,
    ) -> str:
        downloaded.append(origin)
        return str(tmp_path / f'{name}.tif')

    monkeypatch.setattr(cmd, '_download_cityjson_tile', download_tile)

    paths = cmd.load_tile_in_cityjson(
        str(tmp_path),
        'city.json',
        size=40,
        spacial_resolution=0.3,
    )

    assert len(downloaded) == 9
    assert all(path.endswith('.tif') for path in paths)


def test_load_tile_in_cityjson_reports_processing_and_skipping(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cmd.cjio, 'read_cityjson', lambda _: _City())
    monkeypatch.setattr(
        cmd,
        '_download_cityjson_tile',
        lambda *args: str(tmp_path / 'downloaded.tif'),
    )
    existing = tmp_path / 'orthophoto'
    existing.mkdir()
    existing_path = existing / f'40_0_80 ({date.today().isoformat()}).tif'
    existing_path.touch()

    cmd.load_tile_in_cityjson(
        str(tmp_path),
        'city.json',
        size=40,
        verbose=True,
    )

    output = capsys.readouterr().out
    assert 'Processing tile:' in output
    assert f'Skipping tile: 40_0_80 ({date.today().isoformat()})' in output


def test_download_with_retry_reports_failure_and_retry(
    monkeypatch,
    capsys,
) -> None:
    attempts = 0

    def loader() -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError('temporary failure')
        return object()

    monkeypatch.setattr(cmd.time, 'sleep', lambda _: None)

    result = cmd._download_with_retry(loader, 'imagery', True)

    assert result is not None
    assert attempts == 2
    assert 'Failed imagery; retrying in 10 seconds.' in capsys.readouterr().out
