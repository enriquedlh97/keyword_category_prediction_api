#!/bin/bash

if [ "$1" == "c" ]; then
    echo Installing cross-platform environment...
    conda env create -f environment_no_builds.yml ;

elif [ "$1" == "w" ]; then
  echo Installing windows environment...
  conda env create -f environment.yml ;
fi

if [ "$2" == "d" ]; then
  source activate keyword_api ;
  python bin/download_dataset.py ;
fi