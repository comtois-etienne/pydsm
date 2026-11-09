"""
Commands for processing geotiff files
"""


import os

import numpy as np
import pycityjson.io as cjio
from osgeo import gdal
from pycityjson.model import CityObject, GeometryPrimitive
from pycityjson.model.primitive import MultiLineString, MultiPoint, MultiSurface
from pycityjson.model.primitive import Point as CityPoint

import pydsm.geo as geo
import pydsm.nda as nda
import pydsm.rbc as rbc
import pydsm.utils as utils
from pydsm import const


from pydsm.cmd import _print


def generate_rbc_wavefront(
        masks_path: str, 
        ndsm_path: str, 
        dtm_path: str | None,
        ortho_path: str,
        save_dir: str = None, 
        elevation_offset: float = False, 
        verbose: bool = False
    ):
    """
    Generates geometry objects for trees from the masks and the ndsm.  
    The triangulated rhombicuboctahedron (RBC) geometry is used to represent the trees. 
    The dtm is used to get the ground level of the trees.
    The orthophoto is used to get the rejection region of the trees. 

    :param masks_path: str, path to the instance masks (mandatory)
    :param ndsm_path: str, path to the nDSM (mandatory)
    :param dtm_path: str, path to the DTM (optional; defaults to zero elevation)
    :param ortho_path: str, path to the orthophoto (mandatory)
    :param save_dir: str, path to directory to save the wavefront and metadata (optional) (default: current directory)
    :param elevation_offset: float, elevation offset for the trees instead of relying on the DTM (optional) (default: False)
    :param verbose: bool, verbose mode to display the whole process (optional) (default: False)
    :return: None (saves the wavefront and metadata to disk)
    """
    save_dir = save_dir or './'
    if not os.path.exists(save_dir):
        _print(f'* Creating directory \'{save_dir}\'')
        os.makedirs(save_dir)

    _print(f'* Generating wavefront objects from masks in {masks_path}')
    instance_masks = nda.read_numpy(masks_path, npz_format='napari')
    ndsm = geo.open_geotiff(ndsm_path)
    rejection_region = geo.get_outside_mask(geo.open_geotiff(ortho_path))
    dtm = geo.open_geotiff(dtm_path) if dtm_path else 0.0
    uuid = os.path.splitext(os.path.basename(ortho_path))[0]

    wavefront_objects, objects_metadata = rbc.tree_modeling(
        ndsm=ndsm,
        dtm=dtm,
        instance_masks=instance_masks,
        rejection_region=rejection_region,
        uuid=uuid,
        elevation_offset=elevation_offset,
        verbose=verbose,
    )

    obj_path = utils.append_file_to_path(save_dir, f'{uuid}_trees.obj')
    meta_path = utils.append_file_to_path(save_dir, f'{uuid}_trees_metadata.json')
    
    city = cjio.City()
    for wavefront_object in wavefront_objects:
        city["cityobjects"].add_cityobject(wavefront_object)
    cjio.write_as_wavefront(city, obj_path, as_one_geometry=False, swap_yz=False)
    _print(f'* Saved to \'{obj_path}\'')

    utils.write_dict_as_json(objects_metadata, file_path=meta_path)



def generate_rbc_wavefront_tiles(
        tiles_dir: str,
        save_dir: str,
        elevation_offset: float | bool = False,
        verbose: bool = False,
        skip_processed: bool = False,
    ) -> None:
    """Generate RBC Wavefront files for every complete tile in a dataset.

    The dataset must contain the mandatory subdirectories defined in
    :mod:`pydsm.const`. The optional DTM directory may be absent or empty.
    Files are paired by their tile identity. When ``ignore_processed`` is
    true, tiles with both expected output files already in ``save_dir`` are
    skipped.
    """
    required_directories = (
        const.ORTHOPHOTO_SUBDIR,
        const.NDSM_SUBDIR,
        const.INSTANCE_LABELS_SUBDIR,
        const.SEMANTIC_POINTS_SUBDIR,
    )
    directories = {
        name: os.path.join(tiles_dir, name)
        for name in required_directories
    }
    missing_directories = [
        path for path in directories.values() if not os.path.isdir(path)
    ]
    if missing_directories:
        raise FileNotFoundError(
            f"Missing tile directories: {', '.join(missing_directories)}"
        )

    os.makedirs(save_dir, exist_ok=True)
    ndsm_files = sorted(
        os.path.join(directories[const.NDSM_SUBDIR], filename)
        for filename in os.listdir(directories[const.NDSM_SUBDIR])
        if filename.lower().endswith(('.tif', '.tiff'))
    )
    if not ndsm_files:
        raise FileNotFoundError(
            f"No nDSM GeoTIFF files found in {directories[const.NDSM_SUBDIR]}"
        )

    def tile_identity(path: str) -> str:
        return os.path.splitext(os.path.basename(path))[0].split(' (')[0]

    def find_tile_file(directory: str, filename: str, suffixes: tuple[str, ...]) -> str | None:
        exact = os.path.join(directory, filename)
        if os.path.isfile(exact):
            return exact
        identity = tile_identity(filename)
        candidates = sorted(
            os.path.join(directory, candidate)
            for candidate in os.listdir(directory)
            if candidate.lower().endswith(suffixes)
            and tile_identity(candidate) == identity
        )
        return candidates[0] if candidates else None

    dtm_directory = os.path.join(tiles_dir, const.DTM_SUBDIR)
    for ndsm_path in ndsm_files:
        filename = os.path.basename(ndsm_path)
        stem = os.path.splitext(filename)[0]
        ortho_path = find_tile_file(
            directories[const.ORTHOPHOTO_SUBDIR],
            filename,
            ('.tif', '.tiff', '.png'),
        )
        labels_path = find_tile_file(
            directories[const.INSTANCE_LABELS_SUBDIR],
            filename,
            ('.npz',),
        )
        dtm_path = (
            find_tile_file(dtm_directory, filename, ('.tif', '.tiff'))
            if os.path.isdir(dtm_directory) else None
        )
        missing = [
            name for name, path in (
                (const.ORTHOPHOTO_SUBDIR, ortho_path),
                (const.INSTANCE_LABELS_SUBDIR, labels_path),
            )
            if path is None
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing {', '.join(missing)} input(s) for tile {stem}"
            )

        output_uuid = os.path.splitext(os.path.basename(ortho_path))[0]
        if skip_processed and all(
            os.path.isfile(os.path.join(save_dir, filename))
            for filename in (
                f'{output_uuid}_trees.obj',
                f'{output_uuid}_trees_metadata.json',
            )
        ):
            _print(f'* Skipping already processed tile {output_uuid}')
            continue

        generate_rbc_wavefront(
            masks_path=labels_path,
            ndsm_path=ndsm_path,
            dtm_path=dtm_path,
            ortho_path=ortho_path,
            save_dir=save_dir,
            elevation_offset=elevation_offset,
            verbose=verbose,
        )



def generate_rbc_cityjson_tiles(
        tiles_dir: str,
        save_dir: str,
        elevation_offset: float | bool = False,
        verbose: bool = False,
        skip_processed: bool = False,
    ) -> None:
    """Generate one CityJSON file containing RBC trees for each tile.

    The dataset must contain the mandatory subdirectories defined in
    :mod:`pydsm.const`. The optional DTM directory may be absent or empty.
    Each tree's modeling metadata is stored as CityObject attributes.
    """
    required_directories = (
        const.ORTHOPHOTO_SUBDIR,
        const.NDSM_SUBDIR,
        const.INSTANCE_LABELS_SUBDIR,
        const.SEMANTIC_POINTS_SUBDIR,
    )
    directories = {
        name: os.path.join(tiles_dir, name)
        for name in required_directories
    }
    missing_directories = [
        path for path in directories.values() if not os.path.isdir(path)
    ]
    if missing_directories:
        raise FileNotFoundError(
            f"Missing tile directories: {', '.join(missing_directories)}"
        )

    os.makedirs(save_dir, exist_ok=True)
    ndsm_files = sorted(
        os.path.join(directories[const.NDSM_SUBDIR], filename)
        for filename in os.listdir(directories[const.NDSM_SUBDIR])
        if filename.lower().endswith(('.tif', '.tiff'))
    )
    if not ndsm_files:
        raise FileNotFoundError(
            f"No nDSM GeoTIFF files found in {directories[const.NDSM_SUBDIR]}"
        )

    def tile_identity(path: str) -> str:
        return os.path.splitext(os.path.basename(path))[0].split(' (')[0]

    def find_tile_file(
            directory: str,
            filename: str,
            suffixes: tuple[str, ...],
        ) -> str | None:
        exact = os.path.join(directory, filename)
        if os.path.isfile(exact):
            return exact
        identity = tile_identity(filename)
        candidates = sorted(
            os.path.join(directory, candidate)
            for candidate in os.listdir(directory)
            if candidate.lower().endswith(suffixes)
            and tile_identity(candidate) == identity
        )
        return candidates[0] if candidates else None

    dtm_directory = os.path.join(tiles_dir, const.DTM_SUBDIR)
    for ndsm_path in ndsm_files:
        filename = os.path.basename(ndsm_path)
        stem = os.path.splitext(filename)[0]
        ortho_path = find_tile_file(
            directories[const.ORTHOPHOTO_SUBDIR],
            filename,
            ('.tif', '.tiff', '.png'),
        )
        labels_path = find_tile_file(
            directories[const.INSTANCE_LABELS_SUBDIR],
            filename,
            ('.npz',),
        )
        dtm_path = (
            find_tile_file(dtm_directory, filename, ('.tif', '.tiff'))
            if os.path.isdir(dtm_directory) else None
        )
        missing = [
            name for name, path in (
                (const.ORTHOPHOTO_SUBDIR, ortho_path),
                (const.INSTANCE_LABELS_SUBDIR, labels_path),
            )
            if path is None
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing {', '.join(missing)} input(s) for tile {stem}"
            )

        output_uuid = os.path.splitext(os.path.basename(ortho_path))[0]
        cityjson_path = os.path.join(save_dir, f'{output_uuid}_trees.city.json')
        if skip_processed and os.path.isfile(cityjson_path):
            _print(f'* Skipping already processed tile {output_uuid}')
            continue

        _print(f'* Generating CityJSON objects from masks in {labels_path}')
        instance_masks = nda.read_numpy(labels_path, npz_format='napari')
        ndsm = geo.open_geotiff(ndsm_path)
        rejection_region = geo.get_outside_mask(geo.open_geotiff(ortho_path))
        dtm = geo.open_geotiff(dtm_path) if dtm_path else 0.0
        city_objects, objects_metadata = rbc.tree_modeling(
            ndsm=ndsm,
            dtm=dtm,
            instance_masks=instance_masks,
            rejection_region=rejection_region,
            uuid=output_uuid,
            elevation_offset=elevation_offset,
            verbose=verbose,
        )

        city = cjio.City()
        for city_object in city_objects:
            metadata = objects_metadata.get(city_object.uuid())
            if metadata is not None:
                for key, value in metadata.items():
                    city_object.set_attribute(key, value)
            city["cityobjects"].add_cityobject(city_object)
        city.set_geographical_extent()
        cjio.write_as_cityjson(city, cityjson_path, pretty=True)
        _print(f'* Saved to \'{cityjson_path}\'')



def generate_cityjson_terrain(
        dtm: str | gdal.Dataset,
        spacial_resolution: float = 5,
    ) -> cjio.City:
    """
    Generate one CityJSON GroundSurface model from a DTM.

    :param dtm: path to a DTM GeoTIFF or an open GDAL DTM dataset.
    :param spacial_resolution: target spatial resolution in meters per pixel.
    :return: CityJSON city containing one GroundSurface object.
    """
    if spacial_resolution <= 0:
        raise ValueError('spacial_resolution must be greater than zero')

    return _generate_cityjson_terrain(dtm, spacial_resolution)



def generate_terrain_cityjson_tiles(
        tiles_dir: str,
        save_dir: str = 'ground',
        spacial_resolution: float = 5,
        correct_dtm: bool = False,
        verbose: bool = False,
        skip_processed: bool = False,
    ) -> None:
    """
    Generate one CityJSON GroundSurface file for every DTM tile.

    :param tiles_dir: directory containing the DTM subdirectory.
    :param save_dir: output subdirectory relative to tiles_dir (default: ground).
    :param spacial_resolution: target spatial resolution in meters per pixel.
    :param correct_dtm: whether to correct each DTM before export.
    :param verbose: whether to print each processed tile.
    :param skip_processed: whether to skip existing CityJSON files.
    :return: None.
    """
    dtm_directory = os.path.join(tiles_dir, const.DTM_SUBDIR)
    if not os.path.isdir(dtm_directory):
        raise FileNotFoundError(f'Missing DTM directory: {dtm_directory}')
    dtm_paths = _get_dtm_paths(dtm_directory)
    output_directory = os.path.join(tiles_dir, save_dir)
    os.makedirs(output_directory, exist_ok=True)

    for dtm_path in dtm_paths:
        filename = f'{os.path.splitext(os.path.basename(dtm_path))[0]}.city.json'
        output_path = os.path.join(output_directory, filename)
        if skip_processed and os.path.isfile(output_path):
            continue
        if verbose:
            _print(f'* Generating terrain CityJSON from {dtm_path}')
        city = _generate_terrain_cityjson_tile(
            dtm_path,
            spacial_resolution,
            correct_dtm,
        )
        cjio.write_as_cityjson(city, output_path, pretty=True)



def _generate_terrain_cityjson_tile(
        dtm_path: str,
        spacial_resolution: float,
        correct_dtm: bool,
    ) -> cjio.City:
    """
    Generate one terrain CityJSON model, optionally correcting its DTM.

    :param dtm_path: path to the DTM GeoTIFF.
    :param spacial_resolution: target spatial resolution in meters per pixel.
    :param correct_dtm: whether to correct the DTM before export.
    :return: CityJSON city containing one GroundSurface object.
    """
    dtm = geo.open_geotiff(dtm_path)
    if correct_dtm:
        width = geo.get_size(dtm)[0]
        res = geo.get_shape(dtm)[0]
        decimation = max(1, round(res / (width / spacial_resolution)))
        dtm = geo.correct_dtm(dtm, subsampling_size=decimation, fill_outside_mask=None)
    return generate_cityjson_terrain(dtm, spacial_resolution)



def _get_dtm_paths(dtm_directory: str) -> list[str]:
    """
    Find DTM GeoTIFF files in a directory.

    :param dtm_directory: directory containing DTM files.
    :return: sorted DTM file paths.
    """
    paths = sorted(
        os.path.join(dtm_directory, filename)
        for filename in os.listdir(dtm_directory)
        if filename.lower().endswith(('.tif', '.tiff'))
    )
    if not paths:
        raise FileNotFoundError(f'No DTM GeoTIFF files found in {dtm_directory}')
    return paths



def _generate_cityjson_terrain(
        dtm: str | gdal.Dataset,
        spacial_resolution: float,
    ) -> cjio.City:
    """
    Generate one CityJSON GroundSurface from a DTM.

    :param dtm: path to a DTM GeoTIFF or an open GDAL DTM dataset.
    :param spacial_resolution: target spatial resolution in meters per pixel.
    :return: CityJSON city containing one GroundSurface object.
    """
    dtm_dataset = geo.open_geotiff(dtm) if isinstance(dtm, str) else dtm
    if dtm_dataset is None:
        raise ValueError('dtm must be a readable GDAL dataset or GeoTIFF path')
    if dtm_dataset.RasterCount != 1:
        raise ValueError('dtm must contain exactly one raster band')

    width, height = geo.get_size(dtm_dataset)
    target_shape = (
        max(2, round(height / spacial_resolution)),
        max(2, round(width / spacial_resolution)),
    )
    resized_dtm = geo.resize(
        dtm_dataset,
        target_shape,
        scale=spacial_resolution,
    )
    elevation = np.round(geo.to_ndarray(resized_dtm), 3)
    origin_x, origin_y = geo.get_origin(resized_dtm)
    x_coordinates = np.linspace(
        origin_x,
        origin_x + width,
        target_shape[1],
    )
    y_coordinates = np.linspace(
        origin_y,
        origin_y - height,
        target_shape[0],
    )

    vertices = np.empty((target_shape[0], target_shape[1], 3), dtype=float)
    vertices[:, :, 0] = x_coordinates
    vertices[:, :, 1] = y_coordinates[:, np.newaxis]
    vertices[:, :, 2] = elevation

    faces = []
    for row in range(target_shape[0] - 1):
        for column in range(target_shape[1] - 1):
            cell = vertices[row:row + 2, column:column + 2]
            faces.append(
                MultiLineString(
                    [MultiPoint([
                        CityPoint(*cell[0, 0]),
                        CityPoint(*cell[0, 1]),
                        CityPoint(*cell[1, 1]),
                        CityPoint(*cell[1, 0]),
                        CityPoint(*cell[0, 0]),
                    ])]
                )
            )

    ground_surface = CityObject(
        cityobjects=None,
        type='GroundSurface',
        geometries=[GeometryPrimitive(MultiSurface(faces), lod='2')],
        uuid='ground_surface',
    )
    city = cjio.City()
    city['cityobjects'].add_cityobject(ground_surface)
    city.set_geographical_extent()
    return city


