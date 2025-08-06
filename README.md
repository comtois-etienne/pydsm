# pydsm  

DSM, DTM, nDSM and Ortophoto tools


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

