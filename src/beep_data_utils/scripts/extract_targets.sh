# With this script we can add targets from the original data file to the target.csv file, to later on measure the performance on.
poetry run beep-extract-targets \
    --input-csv "/path/to/data.csv" \
    --output-path "/path/to/targets.csv" \
    --target-columns "cycle_life"