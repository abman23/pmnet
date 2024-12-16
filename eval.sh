#!/bin/bash
python eval.py \
    --data_root 'data/' \
    --network 'pmnet_v3' \
    --model_to_eval 'training_results/config_USC_pmnetV3_V2_epoch200/16_0.0001_0.5_10/model_0.00015.pt' \
    --config 'config_USC_pmnetV3_V2'