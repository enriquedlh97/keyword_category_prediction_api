# Keyword Category Prediction API
API for predicting most likely category tags out of 22 different categories from batches of words in 104 different languages. The model has a 76.33% mean average precision and a 93.89% mean AUC ROC.

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
torch==1.9.0+cu102
torchmetrics 0.5.0         
torchtext 0.10.0         
torchvision 0.10.0+cu102
fastapi 0.68.1
pydantic 1.8.2
uvicorn 0.15.0
bayesian-optimization=1.1.0
```
To do this you can create the conda environment and install them directly, or you can just use the .yml files. If you 
are using windows you can just run the following.
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
`string` after the `-d` flag for the actual text you want to predict categories for. 
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
to also test the uncased version as well as other models such as xlm roberta. Within the `cased` directory, the 
`data_module.py` contains the pytorch-lightning wrapper for managing the training, validation and testing datasets via 
the corresponding dataloaders. The `metrics.py` contains the functions for computing the Mean Average Precision of 
the model, the Average Precision for each category, the Mean AUC ROC and the AUC ROC for every category. The `model.py`
file contains the actual implementation of the transformer model as an instance of the pre-trained 
`bert-base-multilingual-cased` model. The `preprocessing.py` file contains all the functions necessary for reading the 
data sets and processing into the input expected by the Dataset module. Finally, the text_dataset contains the logic for
processing the dataset after being preprocessed and tuning it into the input expected by the model. 

The `config.json` file contains the values needed for configuring the pre-trained model when being used by the actual 
API. 

The `bert_final_model_training.ipynb` and `final_model_training.py` contain the logic por actually training the model. 
The first one is a jupyter notebook, and the script version. The actual model was trained with the script version. 
## Pre-trained model details

### Model used

The pre-trained model that was fine-tuned for this specific task was the `bert-base-multilingual-cased`, all the details
about this model can be found [here](https://huggingface.co/bert-base-multilingual-cased). In summary, the model was 
trained on the top 104 languages with the largest Wikipedia using a masked language modeling (MLM) objective. See the 
[list of languages](#list-of-languages) that the Multilingual model supports.

### Fine-tuning

The model was fine-tuned for this specific task by training it on 
[this](https://drive.google.com/uc?id=1LtrGndz9P766BRPf-jWkRw0_gzDuVCVo) dataset for 35 epochs (around 20 hours). A final
linear layer and a sigmoid activation function was added on top of the pre-trained bert model.

Furthermore, a maximum sequence length of 40 tokens was used since the actual maximum length of a sequence in the dataset 
is 33 tokens. 

<p align="center">
  <img src="https://github.com/enriquedlh97/keyword_category_prediction_api/blob/main/token_count.JPG" width="600">

The model was trained for 35 epochs with early stopping (training stopped at epoch 8), a batch size of 64 and a 
learning rate of `2e-5`. No hyperparameter tuning was done due to the time limitations. However, a branch named 
`hyperparam-opt` was created where a hyperparameter optimization subroutine was started to being set up. The intention 
was to use Bayesian Optimization to tune the hyperparameters following a 10-fold cross validation scheme. 

The model achieved a Mean Average Precision of 76.33%. This result is most likely explained by the fact that no 
hyperparameter tuning was done. Since the model was stopped early at epoch 8 even though it had been set up to train for 35 
epochs, it clearly started overfitting after that epoch. 

#### Hardware

The model was fine-tuned for about 5 hours on a server with the following characteristics. 
```text
2 x Intel Xeon Gold 5122 Processor @3.6Ghz (2s, 4c/s, 2t/c = 16 logical CPUs) with 128 GB RAM
1 x Tesla V100-PCIE 32 GB GPU RAM
```
#### AUC ROC
As a reference, after the first epoch the results for this metric were the following.
```text
Mean AUC ROC:0.9118656516075134

AUC ROC per category:

    Health:0.92984116
    Vehicles:0.9358579
    Hobbies & Leisure:0.8849387
    Food & Groceries:0.94045573
    Retailers & General Merchandise:0.8963138
    Arts & Entertainment:0.90174246
    Jobs & Education:0.9257699
    Law & Government:0.91081136
    Home & Garden:0.92352605
    Finance:0.9282235
    Computers & Consumer Electronics:0.9246343
    Internet & Telecom:0.9001928
    Sports & Fitness:0.9006619
    Dining & Nightlife:0.9466539
    Business & Industrial:0.8585652
    Occasions & Gifts:0.9190705
    Travel & Tourism:0.92542875
    News, Media & Publications:0.8619314
    Apparel:0.9306651
    Beauty & Personal Care:0.91301185
    Family & Community:0.86807954
    Real Estate:0.9346674
```

The final results after the complete tuning for 8 epochs are the following. 
```text
Mean AUC ROC:0.9389623999595642

AUC ROC per category:

Health:0.95061696
Vehicles:0.9570166
Hobbies & Leisure:0.9193139
Food & Groceries:0.9607837
Retailers & General Merchandise:0.9425981
Arts & Entertainment:0.924951
Jobs & Education:0.9459585
Law & Government:0.9363848
Home & Garden:0.94838923
Finance:0.95139754
Computers & Consumer Electronics:0.9469222
Internet & Telecom:0.9271123
Sports & Fitness:0.93298805
Dining & Nightlife:0.96544385
Business & Industrial:0.89647937
Occasions & Gifts:0.9483903
Travel & Tourism:0.94743305
News, Media & Publications:0.8890147
Apparel:0.9584558
Beauty & Personal Care:0.9441371
Family & Community:0.9070463
Real Estate:0.9563372
```

#### Mean Average Precision

As a reference, after the first epoch the results for this metric were the following.
```
Mean Average Precision:0.686751127243042

Average Precision per category:

    Health:0.7492238
    Vehicles:0.75872535
    Hobbies & Leisure:0.7025987
    Food & Groceries:0.72701
    Retailers & General Merchandise:0.5111446
    Arts & Entertainment:0.8015246
    Jobs & Education:0.7462203
    Law & Government:0.6179674
    Home & Garden:0.7230513
    Finance:0.64043856
    Computers & Consumer Electronics:0.744525
    Internet & Telecom:0.60483825
    Sports & Fitness:0.67073786
    Dining & Nightlife:0.694178
    Business & Industrial:0.7201561
    Occasions & Gifts:0.6067921
    Travel & Tourism:0.7442349
    News, Media & Publications:0.70804906
    Apparel:0.7202701
    Beauty & Personal Care:0.63098514
    Family & Community:0.5900509
    Real Estate:0.6958019
```

The final results after the complete tuning for 8 epochs are the following. 
```text
Mean Average Precision:0.7633433938026428

Average Precision per category:

Health:0.8023483
Vehicles:0.8293877
Hobbies & Leisure:0.78291553
Food & Groceries:0.79903966
Retailers & General Merchandise:0.65486777
Arts & Entertainment:0.8438803
Jobs & Education:0.7987087
Law & Government:0.68738794
Home & Garden:0.7966564
Finance:0.7317461
Computers & Consumer Electronics:0.8122689
Internet & Telecom:0.69790566
Sports & Fitness:0.75740767
Dining & Nightlife:0.7746685
Business & Industrial:0.79147625
Occasions & Gifts:0.69419265
Travel & Tourism:0.8009785
News, Media & Publications:0.7586461
Apparel:0.8092545
Beauty & Personal Care:0.7334262
Family & Community:0.6742519
Real Estate:0.7621378
```

## Training a model

To train a model from scratch all you have to do is make sure you have the conda environment activated and run the 
`final_model_training.py` script. This script will save the trained model to the assets folder (it will be created if it 
does not already exist). 
```bash
$ python final_model_training.py
```
Currently, the only parameters (and their default values) that can be sepcified for the training subroutine are the following ones.
```text
MAX_TOKEN_COUNT = 40
N_EPOCHS = 20
BATCH_SIZE = 64  
LEARNING_RATE = 2e-5 
DROPOUT = 0.12
```
Each of these parameters can be specified to the `final_model_training.py` script with the flags `--t=MAX_TOKEN_COUNT`, 
`--e=N_EPOCHS`, `--b=BATCH_SIZE`, `--l=LEARNING_RATE` and `--d=DROPOUT`. If no flags are specified the script will run 
with the above mentioned default values. To specify values for these parameters just run something like this example.
```bash
$ python final_model_training.py --t=40 --e=20 --b=64 --l=2e-5 --d=0.12
```
After each epoch, a model is going to be saved in the `assets` folder with the name structure as 
`epoch=EPOCH-val_loss=VALIDATION_LOSS-best-checkpoint.ckpt`. To use one of these models just put the model file name in the `config.json`
file in the `PRETRAINED_MODEL` field. The API will automatically load this model when the server is started. 

## Hyper-parameter optimization
_To be done_

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

The multilingual model supports the following languages. 

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
*   Volapük
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
