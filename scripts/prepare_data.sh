#!/bin/sh

cd sfm &&
python reconstruction.py -d ../data/LEVIR_NVS -s scene_000 -t 1 8 15 --move_to_dataset False ;


