#!/bin/bash

# Path to the pickle files. Pkl file created through these scripts https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/BuildPkl_Batch1.ipynb

poetry run beep-secondary-dataset-creation \
    --pickle-data-path "/path/to/pickle/data"\
    --output-dir "resources" \
    --chunk-size 1000000 \
    --add_cells_to_domain_txt_file False # Needs to be done only once
