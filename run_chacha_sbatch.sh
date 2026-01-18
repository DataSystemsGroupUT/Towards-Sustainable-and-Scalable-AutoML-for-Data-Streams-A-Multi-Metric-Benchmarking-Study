#!/bin/bash
module load miniconda3/22.11.1

eval "$(conda shell.bash hook)"


conda activate oaml_benchmark


# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --dataset_name) dataset_name="$2"; shift ;;
    --seed) seed="$2"; shift ;;
    --CHACHA_ensemble_size) ensemble_size="$2"; shift ;;
  esac
  shift
done

python run_script_2.py \
  --model_name chacha \
  --dataset_name "$dataset_name" \
  --ensemble_size "$ensemble_size" \
  --seed "$seed"