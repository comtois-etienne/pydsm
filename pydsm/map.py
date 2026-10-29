"""Download freely available georeferenced imagery."""

from __future__ import annotations

from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import osgeo
from osgeo import gdal

from . import nda


_WORLD_IMAGERY_EXPORT_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/export"
)


def load_imagery(
    epsg: int,
    origin: tuple[float, float],
    dimension: float = 40,
    pixel_size: float = 1.0,
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
        array,
        epsg=epsg,
        origin=origin,
        pixel_size=actual_pixel_size,
    )