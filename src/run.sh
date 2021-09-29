#!/bin/sh
python src/train.py --model xgb
python src/train.py --model decision_tree
python src/train.py --model rf