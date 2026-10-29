from pathlib import Path

import numpy as np
from osgeo import gdal

from pydsm import geo
from pydsm.map import load_dsm, load_dtm

FIXTURE = Path(__file__).parent / "test_files" / (
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
