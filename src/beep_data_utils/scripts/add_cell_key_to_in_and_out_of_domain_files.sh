# This script can be used to automatically add cells to in and out of domain files

poetry run beep-add-cell-key-to-domains \
    --input-csv "path/to/file.csv" \
    --output-dir "../resources" \
    --test-size 0.1 \
    --append