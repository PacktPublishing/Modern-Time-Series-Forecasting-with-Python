import numpy as np
from darts.models.forecasting.forecasting_model import LocalForecastingModel
from darts import TimeSeries


class NaiveMovingAverage(LocalForecastingModel):
    def __init__(self, window):
        """Naive Mean Model

        This model has no parameter, and always predicts the
        mean value of the training series.
        """
        super().__init__()
        self.window = window
        self.mean_val = None

    def __str__(self):
        return "Naive moving average model"

    def fit(self, series: TimeSeries):
        super().fit(series)
        self._history = series.data_array().to_series().values
        self._fitted_values = (
            series.data_array().to_series().rolling(window=5).mean().bfill()
        )

    def predict(self, n: int, num_samples: int = 1):
        super().predict(n, num_samples)
        history = self._history.copy()
        forecast = np.empty(n)
        for i in range(0, n):
            f = history[-self.window :].mean()
            forecast[i] = f
            history = np.append(history, [f])
        return self._build_forecast_series(forecast)
