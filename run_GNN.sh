#!/bin/bash
DATA="${1-PROTEINS}"
fold=${2-0}  # 0 for 10-fold
GPU=${3-0}

seed=1

CONFIG=configs/${DATA}
if [ ! -f "$CONFIG" ]; then
    echo "No config file for ${DATA} in configs folder"
    exit 128
fi
source configs/${DATA}

FOLDER=results
FILE=${FOLDER}/${DATA}.txt
if [ ! -d "$FOLDER" ]; then
    mkdir $FOLDER
fi

run(){
    CUDA_VISIBLE_DEVICES=${GPU} python3 src/main.py \
        -seed $seed -data $DATA -fold $1 \
        -batch $batch_size -lr $learning_rate \
        -g_conv $graph_conv -g_pool $graph_pool \
        -clusters $num_clusters_layers \
        -ks $pool_rates_layers -acc_file $FILE
}

if [ ${fold} == 0 ]; then
    if [ -f "$FILE" ]; then
        rm $FILE
    fi
    echo "Running 10-fold cross validation"
    start=`date +%s`
    run $fold
    stop=`date +%s`
    echo "End of cross-validation using $[stop - start] seconds"
    echo "The accuracy results for ${DATA} are as follows:"
    cat $FILE
else
    run $fold
    echo "The accuracy result for ${DATA} fold ${fold} is:"
    tail -1 $FILE
fi
