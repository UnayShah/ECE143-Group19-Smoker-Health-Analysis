import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from data_processing import remove_outliers_from_df


def custom_autopct(values: pd.Series) -> any:
    """
    Wrapper function for autopct parameter in pie chart

    Parameters
    ----------
    values : pd.Series[int]
        The values of the categories in the pie chart

    Output
    -------
    any
        The custom autopct function for pie chart
    """
    def my_autopct(pct: float) -> str:
        """
        Custom autopct function for pie chart

        Parameters
        ----------
        pct : float
            The percentage of the value

        Output
        -------
        str
            The formatted string for the percentage and value
        """
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct, v=val)
    return my_autopct


def build_pie_chart(feature: pd.Series, custom_autopct_function: any = None, labels: bool = True) -> None:
    """
    Build a pie chart for a categorical feature

    Parameters
    ----------
    feature : pd.Series
        The feature to be visualized
    custom_autopct_function : any, optional
        The custom autopct function for the pie chart, by default None
    labels : bool, optional
        Whether to show the labels in the pie chart, by default True
    """
    values = feature.value_counts()

    category = feature.value_counts().keys()
    explode_value = [0.1 if values.values[i] == max(
        values.values) else 0 for i in range(len(values))]
    plt.pie(values, labels=category if labels else None, autopct=custom_autopct_function(values) if custom_autopct_function else None,
            shadow=True, explode=explode_value)
    if not labels:
        plt.legend(loc="best",  labels=category)
    plt.title(f'{feature.name}')
    plt.tight_layout(pad=5)


def compare_pie(df1: pd.DataFrame, df2: pd.DataFrame, feature: str) -> None:
    """
    Compare the distribution of a categorical feature in two dataframes using pie chart

    Parameters
    ----------
    df1 : pd.DataFrame
        The first dataframe to be compared
    df2 : pd.DataFrame
        The second dataframe to be compared
    feature : str
        The feature to be compared
    """
    df1_feature = df1[feature]
    df2_feature = df2[feature]

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    build_pie_chart(df1_feature, labels=False)
    plt.subplot(1, 2, 2)
    build_pie_chart(df2_feature, labels=False)
    plt.show()


def build_bar_graph(df: pd.DataFrame, target: str, feature: str) -> None:
    """
    Given a dataframe, build a bar graph for the distribution of a categorical feature with respect to the target

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be visualized
    target : str
        The target column of the dataframe
    feature : str
        The feature to be visualized

    Output
    -------
    None
    """
    unique_values = df[target].unique()
    counts = []
    labels = []
    for value in unique_values:
        counts.append(
            df[df[target] == value][feature].value_counts())
        labels.append(f'{target} = {value}')
    df_feature = pd.DataFrame(counts, index=labels)
    df_feature.plot(kind='bar', stacked=True,
                    title=f'{feature}', width=0.25, figsize=(5, 5))


def build_violin_graph(df: pd.DataFrame, target: str, feature, remove_outliers: bool = False) -> None:
    """
    Given a dataframe, build a violin graph for the distribution of a numerical feature with respect to the target

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be visualized
    target : str
        The target column of the dataframe
    feature : str
        The feature to be visualized
    remove_outliers : bool, optional
        Whether to remove outliers from the dataframe, by default False

    Output
    -------
    None
    """
    df_removed_outliers = remove_outliers_from_df(df, feature, 0.01, 0.99)
    sns.violinplot(x=target, y=feature, data=df_removed_outliers, inner="quart",
                   palette="Set2", hue=target, legend=False)
    plt.title(f'{feature} distribution given {target}', fontsize=12)
    sns.despine(left=True, bottom=True)
    plt.tight_layout(pad=5)


def build_KDE_graph(df: pd.DataFrame, target: str, feature: str, remove_outliers: bool = False) -> None:
    """
    Given a dataframe, build a KDE graph for the distribution of a numerical feature with respect to the target

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be visualized
    target : str
        The target column of the dataframe
    feature : str
        The feature to be visualized
    remove_outliers : bool, optional
        Whether to remove outliers from the dataframe, by default False

    Output
    -------
    None
    """
    if remove_outliers:
        df_removed_outliers = remove_outliers_from_df(df, feature, 0.01, 0.99)
    else:
        df_removed_outliers = df.copy()
    for target_value in df_removed_outliers[target].unique():
        sns.histplot(df_removed_outliers[df_removed_outliers[target] == target_value][feature],
                     label=f'{target} = {target_value}', kde=True, stat='density', bins=10, legend=False).set(ylabel=None)
    plt.title(f'{feature} distribution', fontsize=12)
    plt.xlabel('')
    plt.legend()
    sns.despine(left=True, bottom=True)
    plt.tight_layout(pad=5)


def numerical_correlation(features: pd.DataFrame, color1: str = 'orangered', color2: str = 'darkviolet') -> None:
    """
    Given a dataframe, build a bar graph for the correlation of numerical features with respect to the target

    Parameters
    ----------
    features : pd.DataFrame
        The dataframe to be visualized
    color1 : str, optional
        The color for positive correlation, by default 'orangered'
    color2 : str, optional
        The color for negative correlation, by default 'darkviolet'

    Output
    -------
    None
    """
    plt.figure(figsize=(10, 6))
    features.sort_values().plot(kind='barh', color=np.where(
        features.sort_values().values > 0, color1, color2).T)
    plt.title('Correlation of Numerical Variables with Smoking status')
    plt.xlabel('Correlation')
    plt.show()


def single_feature_plot(df1, df2, x, y, box_title, box_x_label, box_y_label, box_x_ticks, box_x_tick_labels, remove_outliers=False, lower_quartile=0.05, upper_quartile=0.95) -> None:
    """
    For 2 dataframes, plot a box plot and histogram for a single feature.

    Parameters
    ----------
    df1 : pd.DataFrame
        The first dataframe
    df2 : pd.DataFrame
        The second dataframe
    x : str
        The x-axis column name
    y : str
        The y-axis column name
    box_title : str
        The title of the box plot
    box_x_label : str
        The x-axis label of the box plot
    box_y_label : str
        The y-axis label of the box plot
    box_x_ticks : list
        The x-axis ticks of the box plot
    box_x_tick_labels : list
        The x-axis tick labels of the box plot
    remove_outliers : bool, optional
        Whether to remove outliers from the dataframes, by default False
    lower_quartile : float, optional
        The lower quartile value, default is 0.05
    upper_quartile : float, optional
        The upper quartile value, default is 0.95

    Output
    -------
    None
    """
    df1_ = remove_outliers_from_df(
        df1, y, lower_quartile, upper_quartile) if remove_outliers else df1.copy()
    df2_ = remove_outliers_from_df(
        df2, y, lower_quartile, upper_quartile) if remove_outliers else df2.copy()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    sns.boxplot(x=x, y=y, data=df1_)
    sns.boxplot(x=x, y=y, data=df2_)
    plt.xlabel(box_x_label)
    plt.ylabel(box_y_label)
    plt.grid(True)
    plt.title(f'Box Plot {box_title}')
    plt.xticks(box_x_ticks, box_x_tick_labels)

    plt.subplot(1, 2, 2)
    sns.histplot(df1_[y], kde=True, label=box_x_tick_labels[0])
    sns.histplot(df2_[y], kde=True, label=box_x_tick_labels[1])
    plt.xlabel(box_x_label)
    plt.ylabel('Frequency')
    plt.title(f'Histogram {box_title}')
    plt.grid(True)
    plt.legend()
    plt.grid(True)
    plt.show()


def joint_plots(df1, df2, x, y, separation_column, samples=500, remove_outliers=False, lower_quartile=0.05, upper_quartile=0.95) -> None:
    """
    For 2 dataframes, plot a joint and scatter plot for 2 features.

    Parameters
    ----------
    df1 : pd.DataFrame
        The first dataframe
    df2 : pd.DataFrame
        The second dataframe
    x : str
        The x-axis column name
    y : str
        The y-axis column name
    separation_column : str
        The column to separate the dataframes
    samples : int, optional
        The number of samples to be plotted, by default 500
    remove_outliers : bool, optional
        Whether to remove outliers from the dataframes, by default False
    lower_quartile : float, optional
        The lower quartile value, default is 0.05
    upper_quartile : float, optional
        The upper quartile value, default is 0.95

    Output
    -------
    None
    """
    df1_ = remove_outliers_from_df(
        df1, x, lower_quartile, upper_quartile) if remove_outliers else df1.copy()
    df2_ = remove_outliers_from_df(
        df2, x, lower_quartile, upper_quartile) if remove_outliers else df2.copy()
    df1_ = remove_outliers_from_df(
        df1_, y, lower_quartile, upper_quartile) if remove_outliers else df1_.copy()
    df2_ = remove_outliers_from_df(
        df2_, y, lower_quartile, upper_quartile) if remove_outliers else df2_.copy()

    combined_df = pd.concat([df1_, df2_], ignore_index=True)
    title = x + ' Vs ' + y

    combined_df = pd.concat(
        [df1_.sample(samples//2), df2_.sample(samples//2)], ignore_index=True)
    p = sns.jointplot(data=combined_df.sample(samples), x=x, y=y,
                      hue=separation_column, kind='kde', fill=True)
    p.figure.suptitle(title)

    p.ax_joint.collections[1].set_alpha(0.7)
    plt.grid(True)
    p.figure.tight_layout()
    plt.show()
