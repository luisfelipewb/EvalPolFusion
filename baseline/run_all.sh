#!/bin/bash
trap 'echo "Interrupted! Killing all child processes..."; kill 0; exit 1' SIGINT

# Set GPU device here
gpu="cuda:1"

# Hardcoded model types and modalities
model_types=("pspnet" "unet" "deeplab" "segformer")
modalities=("rgb" "dif" "pol")

run_name="norm"
epochs=15

for model in "${model_types[@]}"; do
  for modality in "${modalities[@]}"; do
    combo="--model_type $model --modality $modality"
    echo "Training on $gpu: $combo"
    python train.py --device $gpu --run_name $run_name $combo --epochs $epochs

    echo "Testing on $gpu: $combo"
    python test.py --device $gpu --run_name $run_name $combo
  done
done

echo "All training and test runs completed."
