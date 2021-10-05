#!/bin/sh
python src/training/train.py --model xgb --scale False
python src/training/train.py --model rf --scale False

python src/training/train.py --model linear --scale True
python src/training/train.py --model ridge --scale True

