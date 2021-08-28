# keyword_category_prediction_api
API for predicting categories from batches of words

## Getting Started

To set up a working project there are two possibilities. The automatic easy way and the manual way. Both of them are 
described below. 

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
torch==1.9.0+cu102
torchmetrics 0.5.0         
torchtext 0.10.0         
torchvision 0.10.0+cu102
fastapi 0.68.1
pydantic 1.8.2
uvicorn 0.15.0
```
To do this you can create the conda environment and install them directly, or you can just use the .yml files. If you 
are using windows you can just run the following.
```bash
$ conda env create -f environment.yml
```
If you are not working on Windows then you should use the ```environment_bo_builds.yml``` file instead of the 
```environment.yml```  file for creating the environment because it excludes platform-specific builds. To do this just
run the following. 
```bash
$ conda env create -f environment_bo_builds.yml
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
`string` after the `-d` flag for the actual text you want to predict categories. 
```bash
$ curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "string"
  }'
```
You will get a response containing the probabilities for each category. It will look like this. 
```
{
  "probabilities": {
    "Health": 0.5358596444129944,
    "Vehicles": 0.5567029714584351,
    "Hobbies & Leisure": 0.4562150239944458,
    "Food & Groceries": 0.5050750970840454,
    "Retailers & General Merchandise": 0.46449577808380127,
    "Arts & Entertainment": 0.5480411052703857,
    "Jobs & Education": 0.5760863423347473,
    "Law & Government": 0.498044490814209,
    "Home & Garden": 0.5054681897163391,
    "Finance": 0.5238227844238281,
    "Computers & Consumer Electronics": 0.45684221386909485,
    "Internet & Telecom": 0.5297471880912781,
    "Sports & Fitness": 0.4764178991317749,
    "Dining & Nightlife": 0.45618385076522827,
    "Business & Industrial": 0.4717608392238617,
    "Occasions & Gifts": 0.5496118664741516,
    "Travel & Tourism": 0.437097430229187,
    "News, Media & Publications": 0.5520862340927124,
    "Apparel": 0.5230557322502136,
    "Beauty & Personal Care": 0.5549986362457275,
    "Family & Community": 0.47445714473724365,
    "Real Estate": 0.524471640586853
  }
}
```
The other way is to use the user interface generated by FastAPI. To do this just paste the following address in any 
browser.
```
http://127.0.0.1:8000/docs#/default/predict_predict_post
```
Then, select the `Try it out` button on the top right. Finally, substitute the `"string"` by the actual text you want to 
try and click on `Execute`. 

## Project structure
This project has the following structure
```text
/keyword_category_prediction_api
|-- assets *
|   |-- best-checkpoint.ckpt
|-- bin
|   |-- download_dataset.py
|   |-- download_model.py
|   |-- setup.sh
|   |-- start_server.sh
|-- dataset *
|   |-- keyword_categories
|       |-- keyword_categories
|          |-- keyword_categories.test.jsonl
|          |-- keyword_categories.train.jsonl
|-- keyword_category_predictor
|   |-- models
|   |   |-- __init__.py
|   |   |-- bert_multilingual.py
|   |   |-- model.py
|   |-- __init__.py
|   |-- api.py
|-- modeling
|   |-- bert_base_multilingual
|   |   |-- cased
|   |   |   |-- __init__.py
|   |   |   |-- data_module.py
|   |   |   |-- metrics.py
|   |   |   |-- model.py
|   |   |   |-- preprocessing.py
|   |   |   |-- text_dataset.py
|   |   |-- __init__.py
|   |-- __init__.py
|-- .gitattributes
|-- .gitignore
|-- bert_final_model_training.ipynb
|-- config.json
|-- environment.yml
|-- environment_no_builds.yml
|-- final_model_training.py
|-- README.md
```
Note: The directories marked with `*` are directories that are ignored by git. They will only appear after the set up
script is run. They correspond to the pre-trained model (`assets` directory) and the dataset (`dataset` directory).

Under the `bin` directory are al the scripts necessary for setting up the project and testing the API. 

The `keyword_category_predictor` contains the actual implementation of the API, specifically within the `api.py` file. 
The `keyword_category_predictor/models/bert_multilingual.py` file contains the actual implementation of the model 
instance and the `keyword_category_predictor/models/model.py` file is a wrapper over the model instance to be easily 
used by the API within `api.py` file.

The `modeling` contains the implementation of the actual model that was trained. This directory is structured in such a 
way that more models can be added. Here, the cased version of the bert multilingual model was used, but the intention was 
to also test the uncased version as well as other models such as xlm roberta. Within the `cased` directory, 

## Pre-trained model details

### Model used

### Fine-tuning

#### Methodology

#### Hyperparameters

#### Hardware

#### Mean Average Precision after tuning


## Training a model

## Hyper-parameter optimization
include other models


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
- v22015_runtime=14.27.29016
- win_inet_pton=1.1.0
- wincertstore=0.2
```
Then, add a `#` before each of them. They should end up looking like this. 
```
# - vc=14.2
# - v22015_runtime=14.27.29016
# - win_inet_pton=1.1.0
# - wincertstore=0.2
```

## Resources

The following is a list of some resources used to do this project.

- 
-
-
-
-
-

## Note on .gitignore

The ``` .gitignore ``` file was generated with [gitignore.io](https://www.toptal.com/developers/gitignore) adding the tags for ```python```, ```jupyternotebooks``` and ```pycharm```.
