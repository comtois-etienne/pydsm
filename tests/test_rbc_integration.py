from pathlib import Path

import numpy as np
from osgeo import gdal
from pycityjson.model import CityObject

from pydsm import geo, nda, rbc

TILES = Path(__file__).parent / "test_files" / "tiles"
NDSM_PATH = next((TILES / "ndsm").glob("*.tif"))
DTM_PATH = next((TILES / "dtm").glob("*.tif"))
LABELS_PATH = next((TILES / "labels").glob("*.npz"))


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
