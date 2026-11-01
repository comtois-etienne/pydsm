"""Download freely available georeferenced imagery."""

from __future__ import annotations

from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import osgeo
from osgeo import gdal, osr

from . import nda

_WORLD_IMAGERY_EXPORT_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/export"
)
_HRDEM_WCS_URL = (
    "https://datacube.services.geo.ca/wrapper/ogc/"
    "elevation-hrdem-mosaic"
)
_HRDEM_WCS_EPSG = 3979


def load_imagery(
    epsg: int,
    origin: tuple[float, float],
    dimension: float = 40,
    pixel_size: float = 0.3,
) -> osgeo.gdal.Dataset:
    """Download a square imagery tile as an in-memory GDAL dataset.

    The imagery is requested from Esri's public World Imagery service. The
    origin is the top-left corner in the requested coordinate system and
    ``dimension`` is the width and height in coordinate units (metres for
    projected CRSs). The returned dataset can be passed to
    :func:`pydsm.geo.save_geotiff`.

    :param epsg: EPSG code of the requested coordinate system.
    :param origin: top-left coordinate ``(x, y)`` of the tile.
    :param dimension: tile width and height, in coordinate units.
    :param pixel_size: output pixel size, in coordinate units per pixel.
    :return: an in-memory GDAL dataset.
    """
    if len(origin) != 2:
        raise ValueError("origin must contain exactly two coordinates")
    if dimension <= 0:
        raise ValueError("dimension must be greater than zero")
    if pixel_size <= 0:
        raise ValueError("pixel_size must be greater than zero")

    origin_x, origin_y = origin
    pixels = max(1, round(dimension / pixel_size))
    actual_pixel_size = dimension / pixels
    bbox = (
        origin_x,
        origin_y - dimension,
        origin_x + dimension,
        origin_y,
    )
    query = (
        f"?bbox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        f"&bboxSR={epsg}&imageSR={epsg}"
        f"&size={pixels},{pixels}&format=tiff&f=image"
    )

    try:
        with urlopen(_WORLD_IMAGERY_EXPORT_URL + query, timeout=60) as response:
            image_data = response.read()
    except (HTTPError, URLError) as error:
        raise RuntimeError("Unable to download imagery from the public service") from error

    virtual_path = "/vsimem/pydsm-imagery.tif"
    gdal.FileFromMemBuffer(virtual_path, image_data)
    try:
        source = gdal.Open(virtual_path, gdal.GA_ReadOnly)
        if source is None:
            raise RuntimeError("The imagery service returned an unreadable image")
        array = source.ReadAsArray()
    finally:
        gdal.Unlink(virtual_path)

    if array is None:
        raise RuntimeError("The imagery service returned an empty image")
    if array.ndim == 3:
        array = np.moveaxis(array, 0, -1)

    return nda.to_gdal(
        array[..., :3],
        epsg=epsg,
        origin=origin,
        pixel_size=actual_pixel_size,
    )


def _load_elevation(
    service_url: str,
    coverage: str,
    epsg: int,
    origin: tuple[float, float],
    dimension: float,
    pixel_size: float,
) -> osgeo.gdal.Dataset:
    if not service_url:
        raise ValueError("service_url must not be empty")
    if len(origin) != 2:
        raise ValueError("origin must contain exactly two coordinates")
    if dimension <= 0:
        raise ValueError("dimension must be greater than zero")
    if pixel_size <= 0:
        raise ValueError("pixel_size must be greater than zero")

    origin_x, origin_y = origin
    pixels = max(1, round(dimension / pixel_size))
    target_crs = osr.SpatialReference()
    target_crs.ImportFromEPSG(epsg)
    target_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    wcs_crs = osr.SpatialReference()
    wcs_crs.ImportFromEPSG(_HRDEM_WCS_EPSG)
    wcs_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(target_crs, wcs_crs)
    transformed_corners = [
        transform.TransformPoint(x, y)
        for x, y in (
            (origin_x, origin_y - dimension),
            (origin_x, origin_y),
            (origin_x + dimension, origin_y - dimension),
            (origin_x + dimension, origin_y),
        )
    ]
    wcs_min_x = min(point[0] for point in transformed_corners)
    wcs_max_x = max(point[0] for point in transformed_corners)
    wcs_min_y = min(point[1] for point in transformed_corners)
    wcs_max_y = max(point[1] for point in transformed_corners)
    query = urlencode(
        {
            "service": "WCS",
            "version": "1.1.1",
            "request": "GetCoverage",
            "identifier": coverage,
            "boundingbox": (
                f"{wcs_min_x},{wcs_min_y},{wcs_max_x},{wcs_max_y},"
                f"urn:ogc:def:crs:EPSG::{_HRDEM_WCS_EPSG}"
            ),
            "format": "image/geotiff",
        }
    )

    try:
        with urlopen(f"{service_url}?{query}", timeout=60) as response:
            elevation_data = response.read()
    except (HTTPError, URLError) as error:
        raise RuntimeError(
            f"Unable to download elevation data from {service_url}"
        ) from error

    virtual_path = "/vsimem/pydsm-elevation.tif"
    gdal.FileFromMemBuffer(virtual_path, elevation_data)
    try:
        source = gdal.Open(virtual_path, gdal.GA_ReadOnly)
        if source is None:
            raise RuntimeError("The elevation service returned an unreadable image")
        output = gdal.Warp(
            "",
            source,
            format="MEM",
            dstSRS=f"EPSG:{epsg}",
            outputBounds=(
                origin_x,
                origin_y - dimension,
                origin_x + dimension,
                origin_y,
            ),
            width=pixels,
            height=pixels,
            resampleAlg="near",
            srcNodata=-32767,
            dstNodata=-32767,
        )
    finally:
        gdal.Unlink(virtual_path)

    if output is None:
        raise RuntimeError("The elevation service returned an empty image")
    return output


def load_dtm(
    epsg: int,
    origin: tuple[float, float],
    dimension: float = 40,
    pixel_size: float = 1.0,
    service_url: str = _HRDEM_WCS_URL,
) -> osgeo.gdal.Dataset:
    """Download a digital terrain model as an in-memory GDAL dataset.

    ``service_url`` must point to a WCS that provides terrain elevation.
    """
    return _load_elevation(service_url, "dtm", epsg, origin, dimension, pixel_size)


def load_dsm(
    epsg: int,
    origin: tuple[float, float],
    dimension: float = 40,
    pixel_size: float = 1.0,
    service_url: str = _HRDEM_WCS_URL,
) -> osgeo.gdal.Dataset:
    """Download a digital surface model as an in-memory GDAL dataset.

    ``service_url`` must point to a WCS that provides surface elevation.
    """
    return _load_elevation(service_url, "dsm", epsg, origin, dimension, pixel_size)