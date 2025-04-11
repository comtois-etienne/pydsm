
# 1. reproject geotiff to your local projection
./main.py reproject dsm.tif --save-path 2950/dsm.tif --epsg 2950

# 2. create ndsm from dsm and dtm
./main.py ndsm dsm.tif dtm.tif --correct-dtm --resize orthophoto.tif

# 3.
./main.py zones orthophoto.tif dsm.tif dtm.tif ndsm.tif --save-directory processed/ --translate-file project/2950/translate.csv --geotiff-base-path project/2950/

# 3. crop the original geoTiff from a shapefile (the shapefile can be created from a csv file)
./main.py crop orthophoto.tif square.shp --save-path ortho_croped.tif


./main.py zones data/evelyn_caisse/2950/orthophoto.tif data/evelyn_caisse/2950/ndsm.tif data/evelyn_caisse/2950/dsm.tif data/evelyn_caisse/2950/dtm.tif --save-directory processed/ --translate-file data/evelyn_caisse/2950/translate.csv 