#!/bin/sh
python src/train.py --model xgb --scale False
python src/train.py --model decision_tree --scale False
python src/train.py --model rf --scale False

python src/train.py --model linear --scale True
python src/train.py --model lasso --scale True
python src/train.py --model ridge --scale True
python src/train.py --model svm --scale True
python src/train.py --model svm_linear --scale True
python src/train.py --model svm_poly --scale True
python src/train.py --model svm_sigmoid --scale True
