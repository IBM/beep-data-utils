#!/bin/bash

# Path to the pickle files. Pkl file created through these scripts https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/BuildPkl_Batch1.ipynb

poetry run beep-dataset-creation \
    --pickle-data-path "/path/to/pickle/data" \
    --chunk-size 100000 \
    --target-columns "cycle_life"\
    --output-path "output/path"