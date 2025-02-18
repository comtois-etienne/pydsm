
./main.py --time reproject data/strath/dsm.tif --save-path data/strath/2950/dsm.tif --epsg 2950
./main.py --time reproject data/strath/dtm.tif --save-path data/strath/2950/dtm.tif --epsg 2950
./main.py --time reproject data/strath/orthophoto.tif --save-path data/strath/2950/orthophoto.tif --epsg 2950

./main.py --time ndsm data/strath/2950/dsm.tif data/strath/2950/dtm.tif data/strath/2950/ndsm.tif --correct-dtm

./main.py --time crop data/strath/2950/orthophoto.tif data/strath/square.shp --save-path data/strath/2950/ortho_croped.tif

./main.py --time zones data/strath/2950/orthophoto.tif data/strath/2950/ndsm.tif --save-directory processed