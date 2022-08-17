# Modern-Time-Series-Forecasting-with-Python-
Modern Time Series Forecasting with Python, published by Packt

## Instructions to Setup the Environment
1. Install Anaconda or Miniconda if not done already - `https://www.anaconda.com/products/individual` or `https://docs.conda.io/en/latest/miniconda.html` 
1a. If you are running windows and have not installed Build Tools for Visual Studio in your machine, you need to install that for `fancyimpute` to work. Head over to Additional Installations and complete it.
2. From the root directory of thr repo execute the below command
`conda env create -f anaconda_env.yml`
3. Grab a cup of coffee, cause this can take a while.

### Additional Installations, if needed
If you are on Windows and have not installed Build Tools for Visual Studio in your machine, you need to install that for `fancyimpute` to work.
https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

Select: Workloads → Desktop development with C++

Windows Python needs Visual C++ libraries installed via the SDK to build code, such as via setuptools.extension.Extension or numpy.distutils.core.Extension. For example, building f2py modules in Windows with Python requires Visual C++ SDK as installed above. On Linux and Mac, the C++ libraries are installed with the compiler.
## Instructions to Download Data

1. If you don't have an account at Kaggle, head over to Kaggle and quickly register. https://www.kaggle.com/account/login?phase=startRegisterTab
2. Download the `kaggle.json` and place it in `api_keys` folder and proceed to step 2.
    2a. Go to “Account”, go down the page, and find the “API” section.
    2b. Click the “Create New API Token” button.
    2c. The “kaggle.json” file will be downloaded. Place the file in `api_keys` folder.
3. Activate the anaconda environment - `conda activate modern_ts`
4. Run the following in the anaconda prompt from the root working directory of the Github repo - `python download_data.py`

In case the above is not working for you:

Alternative 1
1. If you don't have an account at Kaggle, head over to Kaggle and quickly register. https://www.kaggle.com/account/login?phase=startRegisterTab
2. Download the `kaggle.json` and place it in `api_keys` folder.
        a. Go to “Account”, go down the page, and find the “API” section.
        b. Click the “Create New API Token” button.
        c. The “kaggle.json” file will be downloaded. Place the file in `api_keys` folder.
3. Run the following command from the root directory of the Github repo project you checked out as part of the environment setup. 
    `kaggle datasets download -d jeanmidev/smart-meters-in-london -p data/london_smart_meters –unzip`

Alternative 2
1. If you don't have an account at Kaggle, head over to Kaggle and quickly register. https://www.kaggle.com/account/login?phase=startRegisterTab
2. Go to https://www.kaggle.com/jeanmidev/smart-meters-in-london and download the dataset
3. Unzip the contents to `data/london_smart_meters`
4. Unzip `hhblock_dataset` to get the raw files we want to work with.

### Final Folder Structure after dataset extraction
```
data
├── london_smart_meters
│   ├── hhblock_dataset
│   │   ├── hhblock_dataset
│   │       ├── hhblock_dataset.csv
│── acorn_details.csv
├── informations_households.csv
├── uk_bank_holidays.csv
├── weather_daily_darksky.csv
├── weather_hourly_darksky.csv
```
# Blocks vs RAM

* 1 or <1 Block for 4GB RAM
* 1 or 2 Blocks for 8GB RAM
* 3 Blocks for 16GB RAM
* 5 Blocks for 32GB RAM
