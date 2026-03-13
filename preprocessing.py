import numpy as np


def dataimport(dataframe, VAR="Value", look_back=10):
    """
    Cleans and smooths a raw time series column.
    Pipeline: outlier removal via rolling mean → linear interpolation → EWMA smoothing.

    Outliers are defined as values deviating more than 5 standard deviations from the local rolling mean.
    """

    # Rolling mean used as local reference for outlier detection
    moving_avg = dataframe[VAR].rolling(window=look_back, min_periods=1).mean()

    threshold = 5 * dataframe[VAR].std()
    outliers = np.abs(dataframe[VAR] - moving_avg) > threshold

    # Replace outliers with NaN and interpolate to preserve temporal continuity
    dataframe.loc[outliers, VAR] = np.nan
    dataframe[VAR] = dataframe[VAR].interpolate()

    # Apply EWMA on the cleaned series to smooth high-frequency noise
    filtered = dataframe[VAR].ewm(span=look_back, adjust=False).mean()

    df_out = filtered.reset_index(drop=True).to_frame(name='Value')
    df_out.index.name = 'Time'
    df_out.reset_index(inplace=True)

    return df_out