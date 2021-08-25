# keyword_category_prediction_api
API for predicting categories from batches of word

## Getting Started

To get a working environment there are two possible options. 

1. Create a conda environment with the listed pre-requisites
2. Create a conda environment from the .yml file


### Pre-requisites

```
python 3.8.5
pytest 6.2.4
```

### Installation

To get started make sure you either have the listed pre-requisites or set up the anaconda environment from the .yml file.

```bash
$ conda env create -f environment.yml
```

Make sure you activate the environment. 
```bash
$ conda activate stanford_algos
```

And verify that it was properly installed.
```bash
$ conda env list
```

Note: If you are not working on Windows then you should use the ```environment_bo_builds.yml``` file instead of the ```environment.yml```  file for creating the environment because it excludes platform-specific builds. 

### Updating environment files
There are two environment files available, the first one is for creating an environment on Windows and the file name is ```environment.yml```. The second one is for crossplatform environments because the packages do not include platform-specific builds, the name of the file is ```environment_no_builds.yml```. Depending on the file that you are going to update the steps to follow are different. 

#### Updating Windows environment file

First make sure the environment is active by running 
```bash
$ conda activate stanford_algos
```

Then after you have updated the environment with some packages, save the new updated environment and override the previous .yml file. To do this, run the following.
```bash
$ conda env export > environment.yml
```

#### Updating Cross-platform environment file

First make sure the environment is active by running 
```bash
$ conda activate stanford_algos
```

Then after you have updated the environment with some packages, save the new updated environment and override the previous .yml file. To do this, run the following.
```bash
$ conda env export -n stanford_algos -f environment_no_builds.yml --no-builds
```

### Setting up the PYTHONPATH

To be able to properly run files or scripts, specially from the terminal or console, make sure to set the PYTHONPATH to include the contents of the directory where the current repository is. 
This needs to be done since this repository is not an installable Python package and there are some modules that need to be imported. 

First, check how the PYTHONPATH is empty by running the following: 
```bash
$ echo $PYTHONPATH
```

To set the PYTHONPATH temporarily for the current terminal session, navigate to the directory containing the repository in your local computer. Once you are in the folder run the following. 
```bash
$ export PYTHONPATH="$PWD"
``` 

You can check how the PYTHONPATH was set correctly by running this command. Note that it should not be empty anymore. 
```bash
$ echo $PYTHONPATH
```

Note: For more information on PYTHONPATH and how to set it permanently see [this](https://bic-berkeley.github.io/psych-214-fall-2016/using_pythonpath.html)

## Tests
For running all tests write:
```
pytest
```
For running some specific tests write (e.g. two_number_sum):
```
pytest tests/divide_conquer_sort_search_random/week_one/integer_multiplication/test_integer_multiplication.py
```
For omitting a particular test, go to the file containing the test and add the following decorator to the test function:
```
@pytest.mark.skip(reason="Should not be run because...")
```

## Note on .gitignore

The ``` .gitignore ``` file was generated with [gitignore.io](https://www.toptal.com/developers/gitignore) adding the tags for ```python```, ```jupyternotebooks``` and ```pycharm```.
