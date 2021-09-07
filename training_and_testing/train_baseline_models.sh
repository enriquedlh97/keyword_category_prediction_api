#!/bin/bash

echo Training Random Forest;
python training_and_testing/train_baseline_model.py --s=0.05 --m='rf';

echo Training Logistic Regression;
python training_and_testing/train_baseline_model.py --s=0.05 --m='lr';

echo Training Support Vector Machine;
python training_and_testing/train_baseline_model.py --s=0.05 --m='svm';