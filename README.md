# Keyword Category Prediction API
API for predicting most likely category tags out of 22 different categories from batches of words in 104 different languages. 
The model has a 76.33% mean average precision, and a 93.89% mean AUC ROC.

## Contents

<!--ts-->
   * [Getting Started](#Getting-Started)
      * [Pre-requisites](#Pre-requisites)
      * [Getting the most recent environment](#Getting-the-most-recent-environment)
   * [API testing](#API-testing)
   * [Setting up the PYTHONPATH](#Setting-up-the-PYTHONPATH)
   * [Models](#Models)
     * [Results and Model Comparison](#Results-and-Model-Comparison)
   * [Model details](#Model-details)
   * [Training and testing models](#Training-and-testing-models)
   * [Hyperparameter optimization](#Hyperparameter-optimization)
   * [Updating environment files](#Updating-environment-files)
<!--te-->

## Getting Started

To set up a working project there are two possibilities. The automatic easy way and the manual way. Both of them are 
described below. Before starting make sure you have `Anaconda` installed, you can download it from [here](https://www.anaconda.com/products/individual).

1. Automatic set up. 

Go into the project directory.
```bash
$ cd keyword_category_prediction_api
```
Grant execution permissions to the set up script. 
```bash
$ chmod u+x bin/setup.sh
```
Finally, run the set up script with the corresponding options `c d m`. 
```bash
$ source bin/setup.sh c d m
```
This script has three options corresponding to three things it does: 
1) Create conda environment 
2) Download dataset
3) Download trained model. 
   
Depending on what you may want to do you can run the 
set up script with different values for these options. For the first option, if you are working on Windows you should 
set it to `w`, otherwise set it to `c`, meaning cross-platform. 

The next option is to download the dataset, which you are going to need if you want to train a model, for this set the 
option to `d`. If you do not want to download the dataset (because you probably do not want to train a model and use 
the pre-trained one instead) set it to `-`.

Finally, the third option downloads the pre-trained model. You should download this model if you want to test the API,
otherwise you will need to first train a model (this can take some time). To download the model set this options to `m`.
If you do not want to download the model just set it to `-`.

1. Manual set up.

First create a conda environment with the following pre-requisites

### Pre-requisites

```
python 3.7.11
pytorch-lightning 1.4.4
transformers 4.3.0
sentencepiece 0.1.96
torch 1.9.0+cu102
torchmetrics 0.5.0         
torchtext 0.10.0         
torchvision 0.10.0+cu102
fastapi 0.68.1
pydantic 1.8.2
uvicorn 0.15.0
bayesian-optimization 1.1.0
scipy 1.5.3
nltk 3.5
```
To do this you can create the conda environment and install them directly, or you can just use the .yml files. If you 
are using windows, then you can just run the following.
```bash
$ conda env create -f environment.yml
```
If you are not working on Windows then you should use the ```environment_no_builds.yml``` file instead of the 
```environment.yml```  file for creating the environment because it excludes platform-specific builds. To do this just
run the following. 
```bash
$ conda env create -f environment_no_builds.yml
```
Make sure you activate the environment. 
```bash
$ conda activate keyword_api
```
Then, download the dataset. This step is optional depending on what you want to do. If you want to train a model you 
should download the dataset. However, if you just want to test the API and do not want to train a model because you want 
to use the pre-trained one instead then you do not need to download the dataset. To download the dataset run the 
following.
```bash
$ python bin/download_dataset.py
```
Finally, download the pre-trained model. This step is also optional. You should download this model if you want to test 
the API, otherwise you will need to first train a model. To download the pre-trained model execute the following command.
```bash
$ python bin/download_model.py
```

### Getting the most recent environment 
If you already have the ```keyword_api``` environment created, and you need to update it because the environment files were 
modified all you have to do is to remove the current environment. To do this first make sure that the environment is not 
currently active. You can just activate the base environment. 
```bash
$ conda activate base
```
Then, you can remove the ```keyword_api``` environment as follows. 
```bash
$ conda remove --name keyword_api --all
```
Finally, run the ```setup.sh``` script just with the flag for the environment. 
```bash
$ source bin/setup.sh c - -
```

## API testing
To test the API you first need to start the app. You can do this by running the start server script. First you should
give execution permission to it. You can do this by running the following command. 
```bash
$ chmod u+x bin/start_server.sh
```
Then you can start the actual app by running the script as follows.
```bash
$ bin/start_server.sh
```
After this you will see that the server is running at `http://127.0.0.1:8000`. To test the actual API there are two 
possibilities. The first one is to test it from the terminal. To do this just use the following command and substitute
`string` values after the `-d` flag for the actual text you want to predict categories for. Make sure you put the text 
inside the brackets since the API takes a list of keywords. This means you can add more than one string. 
```bash
$ curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "batch": ["string 1", "string 2"]
  }'
```
You will get a response containing the probabilities for each category. It will look like this. 
```
{
  "classifications": [
    {
      "keyword": "string 1",
      "labels": [
        {
          "label": "Health",
          "score": 0.011274261400103569
        },
        {
          "label": "Vehicles",
          "score": 0.06771234422922134
        },
        {
          "label": "Hobbies & Leisure",
          "score": 0.195133775472641
        },
        {
          "label": "Food & Groceries",
          "score": 0.0026945434510707855
        },
        {
          "label": "Retailers & General Merchandise",
          "score": 0.0013113673776388168
        },
        {
          "label": "Arts & Entertainment",
          "score": 0.34272900223731995
        },
        {
          "label": "Jobs & Education",
          "score": 0.04811834543943405
        },
        {
          "label": "Law & Government",
          "score": 0.04289892315864563
        },
        {
          "label": "Home & Garden",
          "score": 0.09540806710720062
        },
        {
          "label": "Finance",
          "score": 0.009216787293553352
        },
        {
          "label": "Computers & Consumer Electronics",
          "score": 0.7871538996696472
        },
        {
          "label": "Internet & Telecom",
          "score": 0.5926831960678101
        },
        {
          "label": "Sports & Fitness",
          "score": 0.20651988685131073
        },
        {
          "label": "Dining & Nightlife",
          "score": 0.0017237392021343112
        },
        {
          "label": "Business & Industrial",
          "score": 0.18484708666801453
        },
        {
          "label": "Occasions & Gifts",
          "score": 0.005041724536567926
        },
        {
          "label": "Travel & Tourism",
          "score": 0.044590678066015244
        },
        {
          "label": "News, Media & Publications",
          "score": 0.15313173830509186
        },
        {
          "label": "Apparel",
          "score": 0.03802410140633583
        },
        {
          "label": "Beauty & Personal Care",
          "score": 0.011016074568033218
        },
        {
          "label": "Family & Community",
          "score": 0.06512366980314255
        },
        {
          "label": "Real Estate",
          "score": 0.015770507976412773
        }
      ]
    },
    {
      "keyword": "string 2",
      "labels": [
        {
          "label": "Health",
          "score": 0.00784969236701727
        },
        {
          "label": "Vehicles",
          "score": 0.04988593980669975
        },
        {
          "label": "Hobbies & Leisure",
          "score": 0.14553707838058472
        },
        {
          "label": "Food & Groceries",
          "score": 0.0022167968563735485
        },
        {
          "label": "Retailers & General Merchandise",
          "score": 0.0007766910712234676
        },
        {
          "label": "Arts & Entertainment",
          "score": 0.24853244423866272
        },
        {
          "label": "Jobs & Education",
          "score": 0.027579136192798615
        },
        {
          "label": "Law & Government",
          "score": 0.029029864817857742
        },
        {
          "label": "Home & Garden",
          "score": 0.13453885912895203
        },
        {
          "label": "Finance",
          "score": 0.006180084776133299
        },
        {
          "label": "Computers & Consumer Electronics",
          "score": 0.8416517972946167
        },
        {
          "label": "Internet & Telecom",
          "score": 0.7193861603736877
        },
        {
          "label": "Sports & Fitness",
          "score": 0.17399124801158905
        },
        {
          "label": "Dining & Nightlife",
          "score": 0.0014157937839627266
        },
        {
          "label": "Business & Industrial",
          "score": 0.18167521059513092
        },
        {
          "label": "Occasions & Gifts",
          "score": 0.005364252254366875
        },
        {
          "label": "Travel & Tourism",
          "score": 0.048086315393447876
        },
        {
          "label": "News, Media & Publications",
          "score": 0.11453558504581451
        },
        {
          "label": "Apparel",
          "score": 0.03694669529795647
        },
        {
          "label": "Beauty & Personal Care",
          "score": 0.013184457086026669
        },
        {
          "label": "Family & Community",
          "score": 0.04143988713622093
        },
        {
          "label": "Real Estate",
          "score": 0.011533448472619057
        }
      ]
    }
  ]
}
```
The other way is to use the user interface generated by FastAPI. To do this just paste the following address in any 
browser.
```
http://127.0.0.1:8000/docs#/default/predict_predict_post
```
Then, select the `Try it out` button on the top right. Finally, substitute the `"string"` inside the `[]` by the actual 
text you want to try, and click on `Execute`. Notice that you can add multiple strings by separating them with a comma 
just like in the previous example. For instance, you can try `["string 1", "string 2"]`

## Setting up the PYTHONPATH

To be able to properly run files or scripts, specially from the terminal or console, make sure to set the PYTHONPATH to 
include the contents of the directory where the current repository is. This needs to be done since this repository is 
not an installable Python package and there are some modules that need to be imported. 

First, check how the PYTHONPATH is empty by running the following: 
```bash
$ echo $PYTHONPATH
```

To set the PYTHONPATH temporarily for the current terminal session, navigate to the directory containing the repository 
in your local computer. Once you are in the folder, run the following. 
```bash
$ export PYTHONPATH="$PWD"
``` 

You can check how the PYTHONPATH was set correctly by running this command. Note that it should not be empty anymore. 
```bash
$ echo $PYTHONPATH
```

Note: For more information on PYTHONPATH and how to set it permanently see [this](https://bic-berkeley.github.io/psych-214-fall-2016/using_pythonpath.html)

## Models

This repository implements 4 different models. 

1. BERT base multilingual cased
2. Logistic Regression
3. Support Vector Machine
4. Random Forest

This 4 models were trained and tested with [this](https://drive.google.com/uc?id=1LtrGndz9P766BRPf-jWkRw0_gzDuVCVo) dataset 
using default values for the hyperparameters. Although hyperparameter optimization subroutines were set up for all 
models, they were not run due to time limitations. The API uses the `BERT base multilingual cased` model since it was 
the model that performed the best when comparing the results using default values for all hyperparameters. 

When the results from the hyperparameter optimization subroutines are ready, the four models will be compared once again 
with the optimal values for the hyperparameters and, if necessary, the API will be updated with the best performing model. 

### Results and Model Comparison
The following figures show the comparison of the models. Initially the intention was to use a randomization test to see 
if there was a statistically significant difference among the models, however, since the BERT model ended up performing 
significantly better the tests were not done. 

#### Mean Average Precision and Average Precision per Category
<p align="center">
  <img src="https://github.com/enriquedlh97/keyword_category_prediction_api/blob/main/images/avg_precision_score_results.png" width="600">

<p align="center">
  <img src="https://github.com/enriquedlh97/keyword_category_prediction_api/blob/main/images/mean_avg_precision_results.png" width="600">

#### Mean AUC ROC and AUC ROC per Category
<p align="center">
  <img src="https://github.com/enriquedlh97/keyword_category_prediction_api/blob/main/images/auc_roc_results.png" width="600">

<p align="center">
  <img src="https://github.com/enriquedlh97/keyword_category_prediction_api/blob/main/images/mean_auc_roc_results.png" width="600">

#### Class imbalance and languages

Looking at the results a possible good course of action that could improve the results would be to add additional 
preprocessing steps to handle the category imbalance in the dataset. 

<p align="center">
  <img src="https://github.com/enriquedlh97/keyword_category_prediction_api/blob/main/images/class_distribution.PNG" width="600">

Moreover, preprocessing steps are necessary to handle the different languages present in the dataset. These issues are 
dealt with in the hyperparameter optimization subroutines for all models, however due to the time limitations, the results 
are not ready.

## Model details

### BERT

The pre-trained model that was fine-tuned for this specific task was the `bert-base-multilingual-cased`, all the details
about this model can be found [here](https://huggingface.co/bert-base-multilingual-cased). In summary, the model was 
trained on the top 104 languages with the largest Wikipedia using a masked language modeling (MLM) objective. See the 
[list of languages](#list-of-languages) that the Multilingual model supports.

#### Fine-tuning

The model was fine-tuned for this specific task by training it on the 
[previously](https://drive.google.com/uc?id=1LtrGndz9P766BRPf-jWkRw0_gzDuVCVo) mentioned dataset for 35 epochs. A final
linear layer and a sigmoid activation function were added on top of the pre-trained bert model.

Furthermore, a maximum sequence length of 40 tokens was used since the actual maximum length of a sequence in the dataset 
is 33 tokens. 

<p align="center">
  <img src="https://github.com/enriquedlh97/keyword_category_prediction_api/blob/main/images/token_count.JPG" width="600">

The model was trained for 35 epochs with early stopping (training stopped at epoch 8), a batch size of 64 and a 
learning rate of `2e-5`. These are considered the default values for the hyperparameters. When the results for the 
optimal hyperparameters are ready this file will be updated. 

The model achieved a Mean Average Precision of 76.33%. This result is most likely explained by the fact that no 
hyperparameter tuning was done. Since the model was stopped early at epoch 8 even though it had been set up to train for 35 
epochs, it clearly started overfitting after that epoch.

### Baseline models

The `Logistic Regression`, `Support Vector Machine` and the `Random Forest` models were all trained with the default 
hyperparameters from `scikit-learn`. Additionally, the same vectorizer with the default parameters was used for the three 
models. The vectorizer used was the `TfidfVectorizer` from `scikit-learn`. 

## Training and testing models

#### Hardware

All models were trained on a server with the following characteristics. 
```text
2 x Intel Xeon Gold 5122 Processor @3.6Ghz (2s, 4c/s, 2t/c = 16 logical CPUs) with 128 GB RAM
1 x Tesla V100-PCIE 32 GB GPU RAM
```

### BERT

To fine-tune a BERT model from scratch you have to execute the `training_and_testing/bert_base_multilingual_cased/train.py` 
script. This script takes the following arguments.

- `--t` receives an integer that defines the max token count. The default value is `40`. 
- `--e` receives an integer defining the number of epochs. Defaults to `20`. 
- `--b` receives an integer defining the batch size, defaults to `64`. 
- `--l` receives a float defining the learning rate, defaults to `2e-5`. 
- `--d` receives a float defining the dropout rate, defaults to `0.12`. 
- `--s` receives a float defining the sampling proportion. This useful for running quick tests, and it is not necessary 
  to use all the data. The default value is `1`, meaning 100% of the data.
- `--w` does not receive any value. If the flag is specified then all warnings will be ignored. 
- `--n` Receives a string indicating the name where the training execution will be saved. If nothing is specified, then a 
  folder will be created using the current datetime.
  
For example, if you want to train a model for one epoch using all default values you can run the following. 
```bash
$ python training_and_testing/bert_base_multilingual_cased/train.py --e=1
```
When the script finishes executing, the results will be saved to the `assets/bert_final_training/{specified folder name}` folder,
where the `{specified folder name}` will be the name passed as the `--n` argument, or the current datetime in case nothing was 
specified. 

For example, for a particular datetime the directory and the model file name will look like this.
```text
/keyword_category_prediction_api
|-- assets
|   |-- bert_final_model_training
|       |-- 2021-09-08_11-41-57.089601/
|           |-- epoch=1-val_loss=0.0123-best-checkpoint.ckpt


```

The testing is very similar, for this case you have to execute the `training_and_testing/bert_base_multilingual_cased/test.py` 
script. However, this script does require some arguments indicating the location of the model to be tested. The complete 
set of arguments that this script takes are the following ones. 

- `--s` corresponds to the same argument as in the `train.py` script. 
- `--w` corresponds to the same argument as in the `train.py` script. 
- `--n` receives string corresponding to the name of the directory of training execution where all model data is. 
  This is a required argument.
- `--p` receives string corresponding to the name of the `.ckpt` file of the model to be tested. 
- `--v` receives an integer corresponding to the verbosity. If it is set to `1`, then the metrics are printed at the end. 
  Defaults to `0`. 
  
If you wanted to test the model from the previous example  `epoch=10-val_loss=0.0123-best-checkpoint.ckpt`. You would have 
to run the script with the following arguments. 
```bash
$ training_and_testing/bert_base_multilingual_cased/test.py --n="2021-09-08_11-41-57.089601" --p="epoch=1-val_loss=0.0123-best-checkpoint.ckpt"
```

### Baseline models
The training and testing of the baseline models is very simple and follows the same procedure. 
Within the `training_and_testing` directory, there is a folder for each of the baseline models. Each these folders contains a
hyperparameters.json file that specifies the specific hyperparameters used for that model. 
```text
/keyword_category_prediction_api
|-- training_and_testing
|   |-- logistic_regression
|   |   |--hyperparameters.json
|   |-- random_forest
|   |   |--hyperparameters.json
|   |-- support_vector_machine
|       |--hyperparameters.json
```
All models can be trained using the same `training_and_testing/train_baseline_model.py` script by specifying the 
appropriate arguments. The available arguments are the following ones. 

- `--d` if this flag is specified the model will be trained using the optimal hyperparameters, otherwise the default 
values will be used. At this point it should not be used because the optimal parameters are not available. 
- `--s` receives a float defining the sampling proportion. This useful for running quick tests, and it is not necessary 
  to use all the data. The default value is `1`, meaning 100% of the data.
- `--m` receives a string, `lr` will train the logistic regression, `rf` will train the random forest and `svm` the support
  vector machine.
- `--w` does not receive any value. If the flag is specified then all warnings will be ignored. 

For example, to train the `Logistic Regression` model you can execute. 
```bash
$ python training_and_testing/train_baseline_model.py --m="lr"
```

If you want to train all models at the same time (Logistic Regression, SVM and Random Forest) execute the following command.
```bash
$ training_and_testing/train_baseline_models.sh
```

To test a single model once it has been trained you can do the same but this time with the `test_baseline_model.py` script. 
This script takes following arguments (essentially the same ones as the training script). 
- `--s` receives a float defining the sampling proportion. This useful for running quick tests, and it is not necessary 
  to use all the data. The default value is `1`, meaning 100% of the data.
- `--m` receives a string, `lr` will train the logistic regression, `rf` will train the random forest and `svm` the support
  vector machine.
- `--w` does not receive any value. If the flag is specified then all warnings will be ignored. 
For example, to test a `Logistic Regression` model you can execute the following command. 
```bash
$ python training_and_testing/test_baseline_model.py --m="lr" --s=1
```

Similarly to the training procedure, to test all models simultaneously you can run the following script. 
```bash
$ training_and_testing/test_baseline_models.sh
```

### Train and test all baseline models

To train and test all baseline models simultaneously just execute te `training_and_testing/train_test_baseline_models.sh`
script. 

At the end of the training and testing, all results and models will be found in the `assets/` directory under the 
corresponding model name. For example, the results for `Logistic Regression` will be found at `assets/logistic_regression`.

After each training or testing execution, all results (including AUC and Average Precision metrics as well as the trained 
models) will be saved in the `assets` folder under the corresponding model name.
```text
/keyword_category_prediction_api
|-- assets
|   |-- logistic_regression
|   |-- random_forest
|   |-- support_vector_machine
```
## Hyperparameter optimization
A hyperparameter optimization subroutine was set up for all models using Bayesian Optimization following a 5-fold cross 
validation scheme. These subroutines find optimal hyperparameters for all the models as well as the vectorizers to use 
and the optimal parameters for those vectorizers.  All the scripts for finding the optimal hyperparameters are under the 
`hyperparameter_optimization directory`. There a folder for each model can be found containing a `.py` script corresponding 
to the hyperparameter optimization subroutine and a `.ipynb` file that should be run after the hyperparameter optimization 
subroutine is finished. This latter file will display the optimal hyperparameters as well as a visualization of the gaussian 
process for finding the optimal hyperparameters. 

All hyperparameter optimization scripts follow the same naming structure `hyperparam_opt_model_name_.py` and they receive 
two arguments that control the Bayesian Optimization scheme. 
- `--i` receives an integer and sets the number of steps for random exploration, defaults to `1`. 
- `--n` receives an integer and sets the number of steps for bayesian optimization, defaults to `2`.

In the case of the `BERT` model only the `hyperparam_opt_bert_base_multilingual_cased.py` script has to be executed. 
However, in the case of the baseline models the `optimize_hyperparams.sh` script has to run. 

So, if you wanted to execute the hyperparameter optimization subroutine for the `BERT` model with 20 random exploration 
steps and 200 iterations of Bayesian Optimization you should execute the following command. 
```bash
$ python hyperparameter_optimization/bert_base_multilingual_cased/hyperparam_opt_bert_base_multilingual_cased.py --i=20 --n=200
```
However, in the case of a baseline model, for example the  `Logistic Regression` you should execute the following script. 
```bash
$ hyperparameter_optimization/logistic_regression/optimize_hyperparams.sh
```
Additionally, make sure you set the `--i` and `--n` flags to what you need inside this script, they are set to `--i=20` 
and `--n=200` by default. 

After the hyperparameter optimization subroutine finishes you will find a `logs` directory containing the logs from the 
Bayesian Optimization process. This folder is used by the `.ipynb` file to display the optimal hyperparameters and 
visualize the gaussian process. In the case of the logistic regression the final state of the directory will look like 
this.
```text
/keyword_category_prediction_api
|-- assets
|   |-- logistic_regression
|       |-- logs/
|       |-- hyperparam_opt_logistic_regression.py
|       |-- optimal_hyperparams_logistic_regression.ipynb
|       |-- optimize_hyperparams.sh
|   |-- random_forest
|   |-- support_vector_machine
```
For a sample run of the `Logistic Regression` script setting the flags to `--i=3` and `--n=4` the gaussian process 
visualization will look like this.

<p align="center">
  <img src="https://github.com/enriquedlh97/keyword_category_prediction_api/blob/main/images/gaussian_process.PNG" width="600">

## Updating environment files
There are two environment files available, the first one is for creating an environment on Windows and the file name is 
```environment.yml```. The second one is for cross-platform environments because the packages do not include 
platform-specific builds, the name of the file is ```environment_no_builds.yml```. Depending on the file that you are 
going to update the steps to follow are different. 

### Updating Windows environment file

First make sure the environment is active by running 
```bash
$ conda activate keyword_api
```

Then, after you have updated the environment with some packages, save the new updated environment and override the 
previous .yml file. To do this, run the following.
```bash
$ conda env export > environment.yml
```

### Updating Cross-platform environment file

First make sure the environment is active by running 
```bash
$ conda activate keyword_api
```
Then, after you have updated the environment with some packages, save the new updated environment and override the 
previous .yml file. To do this, run the following.
```bash
$ conda env export -n keyword_api -f environment_no_builds.yml --no-builds
```
Finally, go into the `environment_no_builds.yml` and look for the following packages.
```
- vc=14.2
- vs2015_runtime=14.27.29016
- win_inet_pton=1.1.0
- wincertstore=0.2
- m2w64-libwinpthread-git=5.0.0.4634.697f757
- msys2-conda-epoch=20160418
- m2w64-gmp=6.1.0
- m2w64-gcc-libgfortran=5.3.0
- m2w64-gcc-libs-core=5.3.0
- m2w64-gcc-libs=5.3.0
```
Then, add a `#` before each of them. They should end up looking like this. 
```
# - vc=14.2
# - vs2015_runtime=14.27.29016
# - win_inet_pton=1.1.0
# - wincertstore=0.2
# - m2w64-libwinpthread-git=5.0.0.4634.697f757
# - msys2-conda-epoch=20160418
# - m2w64-gmp=6.1.0
# - m2w64-gcc-libgfortran=5.3.0
# - m2w64-gcc-libs-core=5.3.0
# - m2w64-gcc-libs=5.3.0
# - m2w64-gcc-libs=5.3.0
```

## Resources

The following is a list of some resources used to do this project.

- [Venelin Valkov](https://github.com/curiousily)
- [huggingface](https://huggingface.co/)
- [Analizing multilingual data - Kaggle](https://www.kaggle.com/rtatman/analyzing-multilingual-data)
- [Deploy ML models with FastAPI](https://www.youtube.com/watch?v=b5F667g1yCk)
- [Mean Average Precision](https://www.youtube.com/watch?v=FppOzcDvaDI)
- [Mean Average Precision](https://www.youtube.com/watch?v=oz2dDzsbXr8&list=PL1GQaVhO4f_jE5pnXU_Q4MSrIQx4wpFLM&index=7)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
   
## List of Languages

The multilingual BERT model supports the following languages. 

*   Afrikaans
*   Albanian
*   Arabic
*   Aragonese
*   Armenian
*   Asturian
*   Azerbaijani
*   Bashkir
*   Basque
*   Bavarian
*   Belarusian
*   Bengali
*   Bishnupriya Manipuri
*   Bosnian
*   Breton
*   Bulgarian
*   Burmese
*   Catalan
*   Cebuano
*   Chechen
*   Chinese (Simplified)
*   Chinese (Traditional)
*   Chuvash
*   Croatian
*   Czech
*   Danish
*   Dutch
*   English
*   Estonian
*   Finnish
*   French
*   Galician
*   Georgian
*   German
*   Greek
*   Gujarati
*   Haitian
*   Hebrew
*   Hindi
*   Hungarian
*   Icelandic
*   Ido
*   Indonesian
*   Irish
*   Italian
*   Japanese
*   Javanese
*   Kannada
*   Kazakh
*   Kirghiz
*   Korean
*   Latin
*   Latvian
*   Lithuanian
*   Lombard
*   Low Saxon
*   Luxembourgish
*   Macedonian
*   Malagasy
*   Malay
*   Malayalam
*   Marathi
*   Minangkabau
*   Nepali
*   Newar
*   Norwegian (Bokmal)
*   Norwegian (Nynorsk)
*   Occitan
*   Persian (Farsi)
*   Piedmontese
*   Polish
*   Portuguese
*   Punjabi
*   Romanian
*   Russian
*   Scots
*   Serbian
*   Serbo-Croatian
*   Sicilian
*   Slovak
*   Slovenian
*   South Azerbaijani
*   Spanish
*   Sundanese
*   Swahili
*   Swedish
*   Tagalog
*   Tajik
*   Tamil
*   Tatar
*   Telugu
*   Turkish
*   Ukrainian
*   Urdu
*   Uzbek
*   Vietnamese
*   Volap??k
*   Waray-Waray
*   Welsh
*   West Frisian
*   Western Punjabi
*   Yoruba
*   Thai
*   Mongolian

## Note on .gitignore

The ``` .gitignore ``` file was generated with [gitignore.io](https://www.toptal.com/developers/gitignore) adding the 
tags for ```python```, ```jupyternotebooks``` and ```pycharm```.
