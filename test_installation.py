import numpy as np
x = np.arange(5)
import pandas as pd
df = pd.DataFrame(x)
import torch
d = torch.zeros(3)
print(
    f"GPU: {torch.cuda.is_available()} | # of GPU: {torch.cuda.device_count()}| Default GPU Name: {torch.cuda.get_device_name(0)}"
)
import matplotlib.pyplot as plt
import plotly.express as px
import os
import plotly.io as pio
import pandas as pd
from tqdm.autonotebook import tqdm
import missingno as msno
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
import xgboost
import lightgbm
import catboost
import seaborn
import pmdarima as pm
from darts import TimeSeries
from darts.models import NaiveSeasonal
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import RMSE, MAE
print("#"*25+" All Libraries imported without errors! "+"#"*25)
