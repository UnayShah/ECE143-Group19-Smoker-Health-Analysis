import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats


def df_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the duplicate rows and NaN elements in the dataframe

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be cleaned

    Output
    -------
    pd.DataFrame
        The cleaned dataframe
    """
    print('Count of NaN elements in each column of dataframe:')
    print(df.isnull().sum(), '\n')
    print('Count of duplicate rows in dataframe:')
    print(df.duplicated().sum())
    cleaned_df = df.drop_duplicates().dropna()

    return cleaned_df


def add_columns(dfs: list[pd.DataFrame], function, new_column_name) -> tuple[pd.DataFrame]:
    """
    Add new columns to the dataframes

    Parameters
    ----------
    smokers_df : pd.DataFrame
        The dataframe of smokers
    non_smokers_df : pd.DataFrame
        The dataframe of non-smokers
    function : function
        The function to be applied to the dataframes
    new_column_name : str
        The name of the new column

    Output
    -------
    tuple[pd.DataFrame]
        The updated dataframes
    """
    for df in dfs:
        df[new_column_name] = function(df)
    return dfs


def remove_outliers_from_df(df, y, lower_quartile=0.05, upper_quartile=0.95) -> pd.DataFrame:
    """
    Given a dataframe and a column name, remove the outliers from the dataframe

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe
    y : str
        The column name
    lower_quartile : float
        The lower quartile value, default is 0.05
    upper_quartile : float
        The upper quartile value, default is 0.95

    Output
    -------
    pd.DataFrame
        The dataframe with the outliers removed
    """
    # values in 0.05 and 0.95 quantiles are considered as outliers
    return df[(df[y] > df[y].quantile(lower_quartile)) &
              (df[y] < df[y].quantile(upper_quartile))]


def health_score(df: pd.DataFrame, healthy_ranges: dict, max_score: int = 10) -> pd.DataFrame:
    """
    Calculate the health score of the individuals in the dataframe

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe
    healthy_ranges : dict
        The healthy ranges for the metrics
    max_score : int
        The maximum score, default is 10

    Output
    -------
    pd.DataFrame
        The dataframe with the health scores
    """
    df_copy = df.copy()

    for metric, ranges in healthy_ranges.items():

        df_copy[metric + '_score'] = float(max_score)

        if ranges[0] != 0:
            df_copy.loc[df_copy[metric] < ranges[0], metric + '_score'] = max_score*(
                df_copy.loc[df_copy[metric] < ranges[0], metric]-ranges[0])/(ranges[0]-ranges[2])
            df_copy.loc[df_copy[metric] > ranges[1], metric + '_score'] = max_score*(
                ranges[1] - df_copy.loc[df_copy[metric] > ranges[1], metric])/(ranges[3]-ranges[1])
        else:
            df_copy.loc[df_copy[metric] > ranges[1], metric + '_score'] = max_score * \
                (1 - (df_copy.loc[df_copy[metric]
                 > ranges[1], metric]/ranges[1]))

    df_copy['health_score'] = df_copy[[
        metric + '_score' for metric in healthy_ranges]].mean(axis=1)
    return df_copy


def calculate_bmi(df):
    assert 'weight(kg)' in df.columns and 'height(cm)' in df.columns, 'weight(kg) and height(cm) columns are not present in the dataframe'
    return df['weight(kg)'] / (df['height(cm)'] / 100) ** 2


def calculate_eyesight(df):
    assert 'eyesight(left)' in df.columns and 'eyesight(right)' in df.columns, 'eyesight(left) and eyesight(right) columns are not present in the dataframe'
    return (df['eyesight(left)'] + df['eyesight(right)'])/2


def calculate_total_cholestrol(df):
    assert 'LDL' in df.columns and 'HDL' in df.columns, 'LDL and HDL columns are not present in the dataframe'
    return df['LDL'] + df['HDL']
