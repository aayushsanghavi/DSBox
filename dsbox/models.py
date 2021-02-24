from pandas import DataFrame
from pandas_profiling import ProfileReport


class Data:
    """
    DSBox dataclass that stores the dataframe and other attributes used in the process

    Attributes
    ----------
    df : DataFrame
        pandas dataframe of the dataset

    title : str
        text to describe the dataset

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
            profile_report.to_file(output_file=html_output_file)
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
           profile_report.to_file("your_report.json")
       ```
    """

    def __init__(self, df: DataFrame, title: str):
        self.df = df
        self.title = title
        self.profile_report = ProfileReport(df, title=title, explorative=True)
