# pydsm

DSM, DTM, nDSM and Ortophoto tools


# examples

reprojects all the source files to MTM8
```sh
./main.py --time reproject dsm.tif --save-path 2950/dsm.tif --epsg 2950
./main.py --time reproject dtm.tif --save-path 2950/dtm.tif --epsg 2950
./main.py --time reproject orthophoto.tif --save-path 2950/orthophoto.tif --epsg 2950
```

computes the ndsm from the dsm and dtm
```sh
./main.py --time ndsm 2950/dsm.tif 2950/dtm.tif 2950/ndsm.tif
```

crops the orthophoto using a shapefile containing a square representing the surounding streets
```sh
./main.py --time crop 2950/orthophoto.tif square.shp --save-path 2950/ortho_croped.tif
```

# usefull links

- [mapshaper.org](https://mapshaper.org/) can be used to view shapefiles
- [epsg.io](https://epsg.io/map#srs=2950-1946&x&y&z=10&layer=streets) can be used to verify coordinates in your projection system
- [app.geotiff.io](https://app.geotiff.io/load) can be used to view geotiff (orthophoto) on a map

