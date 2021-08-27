#!/bin/bash

if [ "$1" == "cross" ]
  then
    echo Installing cross-platform environment...
    #conda env create -f environment_no_builds.yml ;
else
  echo Installing windows environment...
  #conda env create -f environment.yml ;
fi


# python download_dataset ;