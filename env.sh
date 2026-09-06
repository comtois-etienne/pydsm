conda create -n pydsm python=3.12 opencv=4.11 matplotlib
conda activate pydsm

conda install plotly
conda install pandas
conda install conda-forge::gdal
conda install conda-forge::scipy
conda install conda-forge::scikit-image
conda install conda-forge::shapely
conda install conda-forge::geopandas
conda install conda-forge::osmnx==2.0.7
conda install conda-forge::tensorflow
conda install conda-forge::ultralytics # todo : modify version to x.x.x
conda install -c pytorch torchvision
conda install "numpy<2"