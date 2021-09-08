#!/bin/bash

python training_and_testing/train_baseline_model.py --s=1 --m='rf';

python training_and_testing/train_baseline_model.py --s=1 --m='lr';

python training_and_testing/train_baseline_model.py --s=1 --m='svm';