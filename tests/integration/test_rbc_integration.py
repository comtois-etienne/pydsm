import json
import shutil
from pathlib import Path

import numpy as np
from osgeo import gdal
from pycityjson.model import CityObject

from pydsm import cmd, geo, nda, rbc

TILES = Path(__file__).parents[2] / "tests" / "test_files" / "tiles"
TILE_ID = "40_299360_5036920"
NDSM_PATH = next((TILES / "ndsm").glob(f"{TILE_ID} *.tif"))
DTM_PATH = next((TILES / "dtm").glob(f"{TILE_ID} *.tif"))
LABELS_PATH = next((TILES / "labels").glob(f"{TILE_ID} *.npz"))


def _inputs() -> tuple[gdal.Dataset, gdal.Dataset, np.ndarray, np.ndarray]:
    ndsm = geo.open_geotiff(str(NDSM_PATH))
    dtm = geo.open_geotiff(str(DTM_PATH))
    labels = nda.read_numpy(str(LABELS_PATH), npz_format="napari")
    instance_masks = (labels == 1).astype(np.uint8)
    rejection_region = np.zeros_like(instance_masks)
    assert ndsm is not None
    assert dtm is not None
    return ndsm, dtm, instance_masks, rejection_region


def test_tree_modeling_accepts_scalar_elevation() -> None:
    ndsm, _, instance_masks, rejection_region = _inputs()

    objects, metadata = rbc.tree_modeling(
        ndsm=ndsm,
        dtm=0.0,
        instance_masks=instance_masks,
        rejection_region=rejection_region,
        uuid="integration",
    )

    assert len(objects) == 1
    assert isinstance(objects[0], CityObject)
    assert metadata["integration_1"]["origin_z"] == 0.0


def test_tree_modeling_resizes_same_extent_dtm() -> None:
    ndsm, dtm, instance_masks, rejection_region = _inputs()

    objects, metadata = rbc.tree_modeling(
        ndsm=ndsm,
        dtm=dtm,
        instance_masks=instance_masks,
        rejection_region=rejection_region,
        uuid="integration",
    )

    assert len(objects) == 1
    assert metadata["integration_1"]["origin_z"] > 0
    assert objects[0].geometries


def test_tree_modeling_skips_out_of_bounds_origin(monkeypatch) -> None:
    ndsm, _, instance_masks, rejection_region = _inputs()
    ndsm_array = geo.to_ndarray(ndsm)
    original_get_rbc_mesh = rbc.get_RBC_mesh

    def out_of_bounds_mesh(*args, **kwargs):
        points, triangles, _ = original_get_rbc_mesh(
            ndsm_array,
            np.zeros_like(instance_masks),
            geo.get_scales(ndsm)[0],
        )
        return points, triangles, np.array([ndsm_array.shape[0], 0])

    monkeypatch.setattr(rbc, "get_RBC_mesh", out_of_bounds_mesh)
    objects, metadata = rbc.tree_modeling(
        ndsm=ndsm,
        dtm=0.0,
        instance_masks=instance_masks,
        rejection_region=rejection_region,
        uuid="integration",
    )

    assert objects == []
    assert metadata == {}


def test_public_rbc_functions_process_fixture() -> None:
    ndsm, _, instance_masks, _ = _inputs()
    ndsm_array = geo.to_ndarray(ndsm)

    assert rbc.get_mask_height(instance_masks, ndsm_array) > 0
    assert rbc.get_mask_area(instance_masks, geo.get_scales(ndsm)[0]) > 0
    points, triangles, _ = rbc.get_RBC_mesh(
        ndsm_array,
        instance_masks,
        geo.get_scales(ndsm)[0],
    )
    assert points.shape[1] == 3
    assert triangles.shape[1] == 3


def test_generate_rbc_wavefront_tiles_allows_empty_dtm(tmp_path: Path) -> None:
    tiles = tmp_path / "tiles"
    output = tmp_path / "output"
    for subdirectory in ("orthophoto", "ndsm", "labels", "points"):
        (tiles / subdirectory).mkdir(parents=True)
    (tiles / "dtm").mkdir()

    shutil.copy(NDSM_PATH, tiles / "ndsm" / NDSM_PATH.name)
    shutil.copy(LABELS_PATH, tiles / "labels" / LABELS_PATH.name)
    shutil.copy(
        next((TILES / "orthophoto").glob("*.tif")),
        tiles / "orthophoto" / next((TILES / "orthophoto").glob("*.tif")).name,
    )

    cmd.generate_rbc_wavefront_tiles(str(tiles), str(output))

    assert next(output.glob("*_trees.obj")).stat().st_size > 0
    assert next(output.glob("*_trees_metadata.json")).exists()
    assert not list(output.iterdir())[0].is_dir()

    shutil.rmtree(tiles / "dtm")
    output_without_dtm = tmp_path / "output_without_dtm"
    cmd.generate_rbc_wavefront_tiles(str(tiles), str(output_without_dtm))
    assert next(output_without_dtm.glob("*_trees.obj")).exists()


def test_generate_rbc_wavefront_tiles_ignores_processed_tiles(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tiles = tmp_path / "tiles"
    output = tmp_path / "output"
    for subdirectory in ("orthophoto", "ndsm", "labels", "points"):
        (tiles / subdirectory).mkdir(parents=True)

    shutil.copy(NDSM_PATH, tiles / "ndsm" / NDSM_PATH.name)
    shutil.copy(LABELS_PATH, tiles / "labels" / LABELS_PATH.name)
    orthophoto_path = next((TILES / "orthophoto").glob("*.tif"))
    shutil.copy(orthophoto_path, tiles / "orthophoto" / orthophoto_path.name)

    output.mkdir()
    output_uuid = orthophoto_path.stem
    (output / f"{output_uuid}_trees.obj").touch()
    (output / f"{output_uuid}_trees_metadata.json").touch()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("processed tile was not skipped")

    monkeypatch.setattr(cmd, "generate_rbc_wavefront", fail_if_called)
    cmd.generate_rbc_wavefront_tiles(
        str(tiles),
        str(output),
        skip_processed=True,
    )


def test_generate_rbc_cityjson_tiles_stores_metadata(tmp_path: Path) -> None:
    tiles = tmp_path / "tiles"
    output = tmp_path / "output"
    for subdirectory in ("orthophoto", "ndsm", "labels", "points"):
        (tiles / subdirectory).mkdir(parents=True)

    shutil.copy(NDSM_PATH, tiles / "ndsm" / NDSM_PATH.name)
    shutil.copy(LABELS_PATH, tiles / "labels" / LABELS_PATH.name)
    orthophoto_path = next((TILES / "orthophoto").glob("*.tif"))
    shutil.copy(orthophoto_path, tiles / "orthophoto" / orthophoto_path.name)

    cmd.generate_rbc_cityjson_tiles(str(tiles), str(output))

    cityjson_path = next(output.glob("*_trees.city.json"))
    cityjson = json.loads(cityjson_path.read_text())
    city_objects = cityjson["CityObjects"]
    assert city_objects
    for tree in city_objects.values():
        assert tree["type"] == "SolitaryVegetationObject"
        assert isinstance(tree["attributes"]["mask_id"], int)
        assert "crown_height" in tree["attributes"]
