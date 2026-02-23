#!/bin/bash

# Base directory to search; default to current dir if not provided
BASE_DIR="docs"

# Array to store failed notebooks
failed_notebooks=()

# Loop over notebooks without subshell so arrays persist
while IFS= read -r notebook; do
    echo "🚀 Running notebook: $notebook"

    nb_name="$(basename "$notebook")"
    nb_dir="$(dirname "$notebook")"
    temp_file="${nb_dir}/${nb_name%.ipynb}_executed_tmp.ipynb"

    # Run the notebook and output to the temp file
    jupyter nbconvert \
        --to notebook \
        --execute "$notebook" \
        --output "$(basename "$temp_file")" \
        --output-dir "$nb_dir" \
        --ExecutePreprocessor.timeout=-1

    if [ $? -ne 0 ]; then
        echo "❌ Failed to execute: $notebook"
        echo "   Temp file kept: $temp_file"
        failed_notebooks+=("$notebook")
    else
        echo "✅ Successfully executed: $notebook"
        rm -f "$temp_file"
    fi
done < <(find "$BASE_DIR" -type f -name "*.ipynb" ! -path "*/.ipynb_checkpoints/*")

# Report failures
if [ ${#failed_notebooks[@]} -gt 0 ]; then
    echo
    echo "⚠️  The following notebooks failed (temp files kept for inspection):"
    for nb in "${failed_notebooks[@]}"; do
        echo " - $nb"
    done
    exit 1
else
    echo
    echo "🎉 All notebooks executed successfully. No temp files remain."
fi
