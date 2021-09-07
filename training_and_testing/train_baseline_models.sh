#!/bin/bash

python training_and_testing/train_baseline_model.py --s=0.05 --m='rf';

python training_and_testing/train_baseline_model.py --s=0.05 --m='lr';

python training_and_testing/train_baseline_model.py --s=0.05 --m='svm';