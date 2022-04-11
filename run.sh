#!/bin/bash

for i in {capsule,}
  do
   echo $i
   rm -rf logs/${i}_train.log
   python -u train.py --config configs/config.yaml --dataset_root ../data/MVTec/ --normal_class $i > logs/${i}_train.log
done
