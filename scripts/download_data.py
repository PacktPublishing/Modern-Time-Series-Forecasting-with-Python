import json
import os
from pathlib import Path
import zipfile
# from logger_api import get_logger
import logging
logger = logging.getLogger(__name__)
# logger = get_logger(__name__)

DATA_PATH = Path("data")
KAGGLE_JSON = Path("api_keys/kaggle.json")
IS_KAGGLE_KEY = KAGGLE_JSON.exists()
KAGGLE_API = None

DATASETS = {
    "1C Sales Dataset": {
        "source": "kaggle",
        "name": "competitive-data-science-predict-future-sales",
        "path": "1c_sales_dataset",
        "filename": "competitive-data-science-predict-future-sales.zip",
    },
    "Montreal Bixi Bike Data": {
        "source": "kaggle",
        "name": "supercooler8/bixi-bike-montreal",
        "path": "bixi_bike_data",
        "filename": "bixi-bike-montreal.zip",
    },
    "Turkish Retail Sales": {
        "source": "Kaggle",
        "name": "berkayalan/retail-sales-data",
        "path": "turkish_retail_sales",
        "filename": "retail-sales-data.zip",
    },
    "Sunspot": {
        "source": "Monash Forecasting Repository",
        "url": "https://zenodo.org/record/4654773/files/sunspot_dataset_with_missing_values.zip?download=1",
        "path": "sunspot",
        "filename": "sunspot.zip",
    },
    "Electricity Demand": {
        "source": "Monash Forecasting Repository",
        "url": "https://zenodo.org/record/4656069/files/elecdemand_dataset.zip?download=1",
        "path": "electricity_demand",
        "filename": "electricity_demand.zip",
    },
    "Dominick Sales": {
        "source": "Monash Forecasting Repository",
        "url": "https://zenodo.org/record/4654802/files/dominick_dataset.zip?download=1",
        "path": "dominick_sales",
        "filename": "dominick_sales.zip",
    },
    # "London Smart Meters": {
    #     "source": "Monash Forecasting Repository",
    #     "url": "https://zenodo.org/record/4656072/files/london_smart_meters_dataset_with_missing_values.zip?download=1",
    #     "path": "london_smart_meters",
    #     "filename": "london_smart_meters.zip",
    # },
    "London Smart Meters": {
        "source": "Kaggle",
        "name": "jeanmidev/smart-meters-in-london",
        "path": "london_smart_meters",
        "filename": "smart-meters-in-london.zip",
    },
    "Tourism": {
        "source": "Monash Forecasting Repository",
        "url": "https://zenodo.org/record/4656096/files/tourism_monthly_dataset.zip?download=1",
        "path": "tourism",
        "filename": "tourism.zip",
    },
}


def get_kaggle_username_key(username=None, key=None):
    _authenticate_api = False
    if ("KAGGLE_USERNAME" in os.environ) and ("KAGGLE_KEY" in os.environ):
        logger.info("Kaggle Username and Key already set as environment variables")
        _authenticate_api = True
    elif (username is not None) and (key is not None):
        logger.info("Kaggle Username and Key retrieved from parameters")
        _authenticate_api = True
    elif IS_KAGGLE_KEY:
        with open(KAGGLE_JSON, "r") as f:
            kaggle_dict = json.load(f)
            username = kaggle_dict["username"]
            key = kaggle_dict["key"]
        logger.info("Kaggle Username and Key retrieved from kaggle.json.")
        _authenticate_api = True
    else:
        logger.warning(
            "kaggle.json not found in api_keys folder, username and key is not passed as parameter or is not set as required environment variables"
        )
    return username, key, _authenticate_api


def get_authenticated_kaggle_api(username=None, key=None):
    global KAGGLE_API
    username, key, _authenticate_api = get_kaggle_username_key(username, key)
    if _authenticate_api and KAGGLE_API is None:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        from kaggle.api.kaggle_api_extended import KaggleApi

        KAGGLE_API = KaggleApi()
        KAGGLE_API.authenticate()
    return KAGGLE_API


def _download_competition_dataset(api, dataset_details):
    api.competition_download_files(
        dataset_details["name"],
        path=DATA_PATH / dataset_details["path"],
        quiet=False,
    )


def _download_dataset(api, dataset_details):
    api.dataset_download_files(
        dataset_details["name"],
        path=DATA_PATH / dataset_details["path"],
        quiet=False,
        unzip=True,
    )


def _unzip(path, filename, delete_zip=True):
    with zipfile.ZipFile(
        str(DATA_PATH / path / filename),
        "r",
    ) as zip_ref:
        zip_ref.extractall(DATA_PATH / path)
    if delete_zip:
        (DATA_PATH / path / filename).unlink()


def download_kaggle_dataset(
    dataset_details, username=None, key=None, competition=False
):
    api = get_authenticated_kaggle_api(username, key)
    if api is not None:
        if competition:
            _download_competition_dataset(api, dataset_details)
            logger.info("Donwload completed. Unzipping..")
            _unzip(dataset_details["path"], dataset_details["filename"], delete_zip=True)
        else:
            _download_dataset(api, dataset_details)
    else:
        raise ValueError(
            "Kaggle API wasn't able to authenticate. Please provide username and key. Refer to README for instructions on how to do that."
        )


def _download(url, filename):
    import functools
    import pathlib
    import shutil
    import requests
    from tqdm.auto import tqdm
    
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get('Content-Length', 0))

    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)

    return path

def download_monash_dataset(dataset_details):
    _download(dataset_details["url"],DATA_PATH/dataset_details["path"]/dataset_details["filename"])
    _unzip(dataset_details["path"], dataset_details["filename"], delete_zip=True)


def download_1c_sales(username=None, key=None):
    logger.info("Downloading 1C Sales Dataset...")
    dataset_details = DATASETS["1C Sales Dataset"]
    download_kaggle_dataset(dataset_details, username, key, competition=True)


def download_bixi_bike(username=None, key=None):
    logger.info("Downloading Montreal Bixi Bike Data Dataset...")
    dataset_details = DATASETS["Montreal Bixi Bike Data"]
    download_kaggle_dataset(dataset_details, username, key, competition=False)

def download_london_smart_meters(username=None, key=None):
    logger.info("Downloading London Smart Meters Dataset...")
    dataset_details = DATASETS["London Smart Meters"]
    download_kaggle_dataset(dataset_details, username, key, competition=False)

def download_sunspot():
    logger.info("Downloading Sunspot Dataset...")
    dataset_details = DATASETS["Sunspot"]
    download_monash_dataset(dataset_details)

def download_electricity_demand():
    logger.info("Downloading Electricity Demand Dataset...")
    dataset_details = DATASETS["Electricity Demand"]
    download_monash_dataset(dataset_details)

def download_dominick_sales():
    logger.info("Downloading Dominick Sales Dataset...")
    dataset_details = DATASETS["Dominick Sales"]
    download_monash_dataset(dataset_details)

def download_turkish_sales_data(username=None, key=None):
    logger.info("Downloading Turkish Sales Dataset...")
    dataset_details = DATASETS["Turkish Retail Sales"]
    download_kaggle_dataset(dataset_details, username, key, competition=False)
    
def download_london_smart_meters(username=None, key=None):
    logger.info("Downloading London Smart Meters Dataset...")
    dataset_details = DATASETS["London Smart Meters"]
    download_kaggle_dataset(dataset_details, username, key, competition=False)
    #Supplementary cleanup
    # os.remove(DATA_PATH/dataset_details['path']/"hhblock_dataset.zip")
    # os.remove(DATA_PATH/dataset_details['path']/"halfhourly_dataset.zip")
    # os.remove(DATA_PATH/dataset_details['path']/"daily_dataset.zip")
    # os.remove(DATA_PATH/dataset_details['path']/"daily_dataset.csv.gz")

def download_tourism():
    logger.info("Downloading Tourism Dataset...")
    dataset_details = DATASETS["Tourism"]
    download_monash_dataset(dataset_details)
# download_1c_sales()
# download_bixi_bike()
# download_sunspot()
# download_electricity_demand()
# download_dominick_sales()
# download_turkish_sales_data()
# download_london_smart_meters()
# download_tourism()
if __name__ == "__main__":
    download_london_smart_meters()
