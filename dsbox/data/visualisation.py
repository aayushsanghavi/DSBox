import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt
from dsbox.utils.logging import LoggingLevel, get_logger

logger = get_logger('visualisation', LoggingLevel.DEBUG)


def plot_feature_correlation(df: DataFrame, title: str = "Feature correlation plot"):
    """
    Creates and plots the feature correlation matrix for the given dataset
    :param df: pandas dataframe of the dataset
    :param title: text to describe the dataset (default = 'Feature correlation plot')
    """
    logger.debug(f"Creating feature correlation matrix")
    correlation_matrix: DataFrame = df.corr()
    plot = sns.heatmap(correlation_matrix,
                       annot=True, fmt='.2f',
                       linewidths=0.5,
                       cbar_kws={'shrink': 0.5},
                       square=True)
    logger.debug(f"Created feature correlation matrix plot")
    sns.despine()
    plot.figure.set_size_inches(14, 12)
    logger.debug(f"Displaying feature correlation matrix plot")
    plt.title(title)
    plt.show()
