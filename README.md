# pydsm  

DSM, DTM, nDSM and Ortophoto tools

## Development environment

Create the project virtual environment with Python 3.12 or newer, then install
the test tools and project dependencies:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[test]"
```

GDAL must be installed by the operating system or environment manager before
installing the Python bindings. On macOS, use a Python environment that
provides matching GDAL bindings (such as a Conda environment) when `pip` cannot
find `gdal-config`.

Run the integration tests and Ruff with:

```sh
.venv/bin/python -m pytest
.venv/bin/ruff check tests
```

For Conda and notebook development, create the named `pydsm` environment from
[`environment.yml`](environment.yml):

```sh
conda env create -f environment.yml
conda run -n pydsm python -m ipykernel install --user \
  --name pydsm --display-name "Python (pydsm)"
conda activate pydsm
python -m pytest
```

Select **Python (pydsm)** as the kernel in an IPython or Jupyter notebook.
The Conda environment installs GDAL before the project dependencies so the
Python GDAL bindings use the matching native library.

Tree modeling is implemented in `pydsm.rbc` and uses
[pycityjson](https://github.com/comtois-etienne/pycityjson) for CityObject
geometry and Wavefront export. The DTM argument to `tree_modeling` may be a
constant elevation or a GDAL dataset covering the same spatial extent as the
nDSM; lower-resolution DTM datasets are resized to the nDSM grid.

To export one CityJSON file per tile, with each tree's modeling metadata stored
on its `SolitaryVegetationObject`, use:

```python
from pydsm.cmd import generate_rbc_cityjson_tiles

generate_rbc_cityjson_tiles(
    tiles_dir="/path/to/tiles",
    save_dir="/path/to/tiles/cityjson",
    skip_processed=True,
)
```

To create a CityJSON `GroundSurface` from one DTM, use:

```python
from pycityjson.io import write_as_cityjson
from pydsm.cmd import generate_cityjson_terrain

terrain = generate_cityjson_terrain("/path/to/dtm.tif")
write_as_cityjson(terrain, "/path/to/terrain.city.json", pretty=True)
```

The default spatial resolution is 2.5 m/px, producing a 16-by-16 terrain
grid for a 40 m tile. Elevations are rounded to millimeters.
To process a folder of DTM tiles, pass the folder path; one
`*_ground.city.json` file is written per tile to `ground` by default:

```python
generate_cityjson_terrain("/path/to/tiles/dtm")
```

Elevation tiles can be downloaded with `pydsm.map.load_dtm()` and
`pydsm.map.load_dsm()`. Both functions return in-memory GDAL datasets that can
be saved with `pydsm.geo.save_geotiff()`. The default source is Natural
Resources Canada's public HRDEM Mosaic WCS, which provides the `dtm` and `dsm`
coverages without a token. A compatible WCS URL can be passed with
`service_url`.


## pipeline example  

1. show info about the geotiff
```sh
./main.py info project/orthophoto.tif
```

2. reproject all the geotiff to your local projection  
```sh
./main.py reproject project/dsm.tif --save-path project/2950/dsm.tif --epsg 2950
./main.py reproject project/dtm.tif --save-path project/2950/dtm.tif --epsg 2950
./main.py reproject project/orthophoto.tif --save-path project/2950/orthophoto.tif --epsg 2950
```

3. compute the ndsm from the dsm and dtm  
```sh
./main.py --time ndsm project/2950/dsm.tif project/2950/dtm.tif project/2950/ndsm.tif
```

4. align the geotiff onto the osm map  
```sh
./main.py registration project/2950/orthophoto.tif --save-path project/2950/translate.csv
```

5. extract zones based on city blocks (may fail if not surrounded by streets) and crops the geotiff with the precalculated registration  
```sh
./main.py zones orthophoto.tif dsm.tif dtm.tif ndsm.tif --save-directory processed/ --translate-file project/2950/translate.csv --geotiff-base-path project/2950/
```


## usefull links  

- [mapshaper.org](https://mapshaper.org/) can be used to view shapefiles
- [epsg.io](https://epsg.io/map#srs=2950-1946&x&y&z=10&layer=streets) can be used to verify coordinates in your projection system
- [app.geotiff.io](https://app.geotiff.io/load) can be used to view geotiff (orthophoto) on a map


## dependency graph  

to avoid circular dependencies, the files can be imported in such a way that :  

`main` --__import__--> (`cmd`, `geo`, `nda`, `shp`, `utils`)  
`cmd` --__import__--> (`geo`, `nda`, `shp`, `utils`)  
`geo` --__import__--> (`nda`, `shp`, `utils`)  
`nda` --__import__--> (`shp`, `utils`)  
`shp` --__import__--> (`utils`)  
`utils`  
