#!/bin/bash

I_F1='/Users/etiennecomtois/Downloads/tiles/dataset_1250_40m_instance/cv/fold_1'
I_F2='/Users/etiennecomtois/Downloads/tiles/dataset_1250_40m_instance/cv/fold_2'
I_F3='/Users/etiennecomtois/Downloads/tiles/dataset_1250_40m_instance/cv/fold_3'
I_F4='/Users/etiennecomtois/Downloads/tiles/dataset_1250_40m_instance/cv/fold_4'
I_F5='/Users/etiennecomtois/Downloads/tiles/dataset_1250_40m_instance/cv/fold_5'

S_F1='/Users/etiennecomtois/Downloads/tiles/dataset_1250_40m_species/cv/fold_1'
S_F2='/Users/etiennecomtois/Downloads/tiles/dataset_1250_40m_species/cv/fold_2'
S_F3='/Users/etiennecomtois/Downloads/tiles/dataset_1250_40m_species/cv/fold_3'
S_F4='/Users/etiennecomtois/Downloads/tiles/dataset_1250_40m_species/cv/fold_4'
S_F5='/Users/etiennecomtois/Downloads/tiles/dataset_1250_40m_species/cv/fold_5'

source venv_ultralytics/bin/activate


python3 ./yolo_tile_validation.py rgbd $I_F5 '/Users/etiennecomtois/Downloads/comet/runs/yolo11l_instance_rgbd_1984_f5_60e_(20260321 23h29)'


# python3 ./yolo_tile_validation.py rgb $I_F1 
# python3 ./yolo_tile_validation.py rgb $I_F2 
# python3 ./yolo_tile_validation.py rgb $I_F3 
# python3 ./yolo_tile_validation.py rgb $I_F4 
# python3 ./yolo_tile_validation.py rgb $I_F5 


