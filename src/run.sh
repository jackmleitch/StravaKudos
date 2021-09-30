#!/bin/sh
python src/train.py --model xgb --scale False
python src/train.py --model rf --scale False

python src/train.py --model linear --scale True
python src/train.py --model ridge --scale True

