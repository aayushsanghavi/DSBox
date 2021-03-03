from pandas import DataFrame
from pandas_profiling import ProfileReport

from dsbox.data.preprocessing import preprocess
from dsbox.data.visualisation import plot_feature_correlation
from dsbox.utils.logging import get_logger, LoggingLevel
from dsbox.data.cleanup import cleanup


class Data:
    """
    DSBox dataclass that stores the dataframe and other attributes used in the process

    Attributes
    ----------
    df : DataFrame
        pandas dataframe of the dataset

    title : str
        text to describe the dataset

        Examples: 'Iris dataset', 'Titanic dataset'

    target : str
        name of the prediction target column

    Methods
    -------
    profile_report : ProfileReport
        pandas_profiling report of the dataframe

        The pandas `df.describe()` method is great but a little basic for serious exploratory data analysis.
        profile_report uses pandas_profiling which provides the following statistics (if relevant for the column type):

        1. Type inference: detect the types of columns in a dataframe.
        2. Essentials: type, unique values, missing values
        3. Quantile statistics like minimum value, Q1, median, Q3, maximum, range, inter quartile range
        4. Descriptive statistics like mean, mode, standard deviation, sum, median absolute deviation, coefficient of
           variation, kurtosis, skewness
        5. Most frequent values
        6. Histograms
        7. Correlations highlighting of highly correlated variables, Spearman, Pearson and Kendall matrices
        8. Missing values matrix, count, heatmap and dendrogram of missing values
        9. Duplicate rows Lists the most occurring duplicate rows
        10. Text analysis about categories (Uppercase, Space), scripts (Latin, Cyrillic) and blocks (ASCII) of text data

        Examples on using the profile_report:

        1. Export the profile_report to html file
        ```
            profile_report.to_file(output_file=profile_report.html)
        ```

        2. Display the profile_report in a notebook
        ```
            profile_report.to_widgets()
       ```

       3. Export profile_report to json file or string
       ```
           # As a string
           json_data = profile_report.to_json()

           # As a file
           profile_report.to_file("profile_report.json")
       ```
    """

    def __init__(self, df: DataFrame, title: str, target: str):
        self.df = df
        self.title = title
        self.target = target
        self.__logger = get_logger('Data', LoggingLevel.DEBUG)

    @property
    def profile_report(self) -> ProfileReport:
        title = f"Profile report : {self.title}"
        self.__logger.debug(f"Creating profile report")
        return ProfileReport(self.df, title=title, explorative=True, progress_bar=False)

    def plot_feature_correlation(self):
        plot_title = f"Feature correlation plot : {self.title}"
        self.__logger.debug(f"Plotting feature correlation")
        plot_feature_correlation(self.df, plot_title)

    def cleanup(self, auto=False):
        self.__logger.debug(f"Starting dataset cleanup. Auto-mode : {auto}")
        cleanup(self.df, auto)
        self.__logger.debug(f"Finished dataset cleanup. Auto-mode : {auto}")

    def preprocess(self, auto=False):
        self.__logger.debug(f"Starting dataset preprocessing. Auto-mode : {auto}")
        preprocess(self.df, auto)
        self.__logger.debug(f"Finished dataset preprocessing. Auto-mode : {auto}")
