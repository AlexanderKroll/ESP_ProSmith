# ProSmith
This repository contains the code and datato execute the trained [ProSmith model for the enzyme-substrate pair prediction task](https://doi.org/10.1101/2023.08.21.554147).

## Downloading data folder
Before you can reproduce the results of the manuscript, you need to [download and unzip a data folder from Zenodo](https://doi.org/10.5281/zenodo.10988282).
Afterwards, this repository should have the following strcuture:

    ├── code
    ├── data   
    ├── LICENSE.md 
    ├── requirements.txt 
    ├── environment.yml     
    ├── Tutorial for running ProSmith ESP.ipynb
    └── README.md

## Install
```
conda env create -f environment.yml
conda activate esp_prosmith
pip install -r requirements.txt
```

## For instructions on how to use the trained ESP ProSmith model, see "Tutorial for running ProSmith ESP.ipynb".


 
