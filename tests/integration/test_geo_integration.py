from pathlib import Path

import numpy as np
from osgeo import gdal
from pycityjson.model import City

from pydsm import cmd, geo
from pydsm.map import load_dsm, load_dtm

FIXTURE = Path(__file__).parents[2] / "tests" / "test_files" / "tiles" / "orthophoto" / (
    "40_299360_5036920 (2023-07-21) (69m).tif"
)


def test_save_geotiff_round_trip_preserves_fixture(tmp_path: Path) -> None:
    source = gdal.Open(str(FIXTURE), gdal.GA_ReadOnly)
    assert source is not None

    output_path = tmp_path / "saved.tif"
    geo.save_geotiff(source, str(output_path))

    saved = gdal.Open(str(output_path), gdal.GA_ReadOnly)
    assert saved is not None
    assert saved.RasterXSize == source.RasterXSize
    assert saved.RasterYSize == source.RasterYSize
    assert saved.RasterCount == source.RasterCount
    assert saved.GetGeoTransform() == source.GetGeoTransform()
    assert geo.get_epsg(saved) == geo.get_epsg(source)

    for band_number in range(1, source.RasterCount + 1):
        np.testing.assert_array_equal(
            source.GetRasterBand(band_number).ReadAsArray(),
            saved.GetRasterBand(band_number).ReadAsArray(),
        )


def test_generate_cityjson_terrain_uses_fixed_grid_and_rounds_values() -> None:
    dtm = next(
        (Path(__file__).parents[2] / "tests" / "test_files" / "tiles" / "dtm").glob(
            "40_300080_5037320 *.tif"
        )
    )

    city = cmd.generate_cityjson_terrain(str(dtm))
    ground_surface = city['cityobjects'].get_by_uuid('ground_surface')
    assert ground_surface is not None
    assert ground_surface.type == 'GroundSurface'
    assert ground_surface.geometries[0].surface_count() == 49
    vertices = np.unique(
        np.asarray(ground_surface.get_vertices()).reshape(-1, 3),
        axis=0,
    )
    assert len(vertices) == 64
    assert not np.any(np.isclose(vertices[:, 2], 0))
    assert all(
        abs(round(vertex[2], 3) - vertex[2]) <= 0.0005
        for vertex in vertices
    )


def test_generate_cityjson_terrain_has_matching_adjacent_boundary_points() -> None:
    dtm_directory = Path(__file__).parents[2] / "tests" / "test_files" / "tiles" / "dtm"
    first = cmd.generate_cityjson_terrain(
        str(next(dtm_directory.glob("40_300080_5037320 *.tif")))
    )
    second = cmd.generate_cityjson_terrain(
        str(next(dtm_directory.glob("40_300120_5037320 *.tif")))
    )

    def vertices(city: City) -> np.ndarray:
        ground_surface = city['cityobjects'].get_by_uuid('ground_surface')
        return np.unique(
            np.asarray(ground_surface.get_vertices()).reshape(-1, 3),
            axis=0,
        )

    first_boundary = vertices(first)
    second_boundary = vertices(second)
    first_boundary = first_boundary[
        np.isclose(first_boundary[:, 0], 300120, rtol=0, atol=0.001)
    ]
    second_boundary = second_boundary[
        np.isclose(second_boundary[:, 0], 300120, rtol=0, atol=0.001)
    ]
    assert {
        tuple(point[:2]) for point in first_boundary
    } == {
        tuple(point[:2]) for point in second_boundary
    }


def test_generate_terrain_cityjson_tiles_processes_dtm_folder(tmp_path: Path) -> None:
    dtm_directory = Path(__file__).parents[2] / "tests" / "test_files" / "tiles" / "dtm"
    input_directory = tmp_path / "tiles"
    dtm_input_directory = input_directory / "dtm"
    output_directory = input_directory / "ground"
    dtm_input_directory.mkdir(parents=True)
    for pattern in ("40_300080_5037320 *.tif", "40_300120_5037320 *.tif"):
        source = next(dtm_directory.glob(pattern))
        source_copy = dtm_input_directory / source.name
        source_copy.write_bytes(source.read_bytes())

    cmd.generate_terrain_cityjson_tiles(
        str(input_directory),
        save_dir='ground',
    )

    output_directory = input_directory / 'ground'
    outputs = sorted(output_directory.glob("*.city.json"))
    assert len(outputs) == 2
    assert all(output.stat().st_size > 0 for output in outputs)


def test_generate_terrain_cityjson_tiles_corrects_dtm_at_target_grid_size(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dtm_directory = tmp_path / 'tiles' / 'dtm'
    dtm_directory.mkdir(parents=True)
    source = next(
        (Path(__file__).parents[2] / 'tests' / 'test_files' / 'tiles' / 'dtm').glob(
            '40_300080_5037320 *.tif'
        )
    )
    dtm_path = dtm_directory / source.name
    dtm_path.write_bytes(source.read_bytes())
    calls: list[int] = []

    def correct_dtm(dataset: gdal.Dataset, subsampling_size: int) -> gdal.Dataset:
        calls.append(subsampling_size)
        return dataset

    monkeypatch.setattr(geo, 'correct_dtm', correct_dtm)
    cmd.generate_terrain_cityjson_tiles(
        str(tmp_path / 'tiles'),
        spacial_resolution=5,
        correct_dtm=True,
    )

    assert calls == [8]


def test_save_geotiff_creates_missing_parent_directory(tmp_path: Path) -> None:
    source = gdal.Open(str(FIXTURE), gdal.GA_ReadOnly)
    assert source is not None

    output_path = tmp_path / "nested" / "saved.tif"
    geo.save_geotiff(source, str(output_path))

    assert output_path.is_file()
    saved = gdal.Open(str(output_path), gdal.GA_ReadOnly)
    assert saved is not None


def test_save_geotiff_uses_strips_for_small_rasters(tmp_path: Path) -> None:
    source = gdal.GetDriverByName("MEM").Create("", 133, 133, 4, gdal.GDT_Byte)
    assert source is not None

    output_path = tmp_path / "small.tif"
    geo.save_geotiff(source, str(output_path))

    saved = gdal.Open(str(output_path), gdal.GA_ReadOnly)
    assert saved is not None
    assert saved.GetMetadataItem("TILED", "IMAGE_STRUCTURE") is None


def test_download_and_save_dtm(tmp_path: Path) -> None:
    dataset = load_dtm(2950, (299360, 5036920), dimension=40, pixel_size=1)
    values = dataset.ReadAsArray()
    assert values is not None
    assert not np.any(values == -32767)
    output_path = tmp_path / "dtm.tif"
    geo.save_geotiff(dataset, str(output_path))

    saved = gdal.Open(str(output_path), gdal.GA_ReadOnly)
    assert saved is not None
    assert (saved.RasterXSize, saved.RasterYSize) == (40, 40)
    assert geo.get_epsg(saved) == 2950


def test_download_and_save_dsm(tmp_path: Path) -> None:
    dataset = load_dsm(2950, (299360, 5036920), dimension=40, pixel_size=1)
    values = dataset.ReadAsArray()
    assert values is not None
    assert not np.any(values == -32767)
    output_path = tmp_path / "dsm.tif"
    geo.save_geotiff(dataset, str(output_path))

    saved = gdal.Open(str(output_path), gdal.GA_ReadOnly)
    assert saved is not None
    assert (saved.RasterXSize, saved.RasterYSize) == (40, 40)
    assert geo.get_epsg(saved) == 2950
