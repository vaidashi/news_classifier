export PYTHONPATH=$PYTHONPATH:$PWD
export EXPERIMENT_NAME="llm"
export DATASET_LOC="data/BBC_News.csv"
export TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'
export INITIAL_PARAMS="[{\"train_loop_config\": $TRAIN_LOOP_CONFIG}]"

python scripts/tune.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --dataset-loc "$DATASET_LOC" \
    --initial-params "$INITIAL_PARAMS" \
    --num-runs 2 \
    --num-workers 1 \
    --cpu-per-worker 1 \
    --gpu-per-worker 0 \
    --num-epochs 10 \
    --batch-size 256 \
    --results-fp results/tuning_results.json