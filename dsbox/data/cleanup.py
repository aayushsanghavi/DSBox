from typing import List
import numpy as np
from pandas import DataFrame

from dsbox.utils.logging import get_logger, LoggingLevel

logger = get_logger('cleanup', LoggingLevel.DEBUG)


def identify_single_value_columns(df: DataFrame) -> List[str]:
    """
    Returns a list of features in the dataset which have only one unique value.
    Such features don't contribute to model learning and should be removed.

    :param df: pandas dataframe of the dataset
    :return: list of columns with only one unique feature value
    """

    return [column for column in df.columns if len(df[column].unique()) == 1]


def __is_identifier_column_name(column_name: str) -> bool:
    return 'id' == column_name.lower() or 'ID' in column_name or 'Id' in column_name or '_id' in column_name


def identify_potential_id_columns(df: DataFrame) -> List[str]:
    """
    Returns a list of features in the dataset which could be potential numeric identifiers.
    Matching is done based on the datatype of the column -> should be int or long
    and the column name should match contain 'ID' or 'Id' or '_id'.
    Identifier features don't contribute to model learning and should be removed.

    :param df: pandas dataframe of the dataset
    :return: list of columns which are identified as potential identifiers
    """

    return [column for column in df.columns
            if df[column].dtypes in (np.int32, np.int64) and __is_identifier_column_name(column)]


def identify_columns_with_majority_missing_values(df: DataFrame, threshold: float = 0.8) -> List[str]:
    """
    Identifies features with majority rows having missing values.
    Such Features tend to be not so important for model learning because of huge chunk of missing values.

    :param df: pandas dataframe of the dataset
    :param threshold: fraction defining the minimum % of missing values in the column
    :return: list of columns which have more missing values than the specified threshold
    """

    missing_values_count = df.isnull().sum().to_dict()
    threshold_count = round(df.shape[0] * threshold)
    logger.debug(f"Threshold count of missing column values : {threshold_count}")
    return [col for col, count in missing_values_count.items() if count >= threshold_count]


def __process_cleanup_action(action: str, column_list: List[str]) -> List[str]:
    """
    Decides how to process the chosen action for cleanup step.
    If DROP_ALL -> return :param column_list.
    If not NO_OP -> parse and return column names.

    :param action: string text representing the action. Options -> DROP_ALL, NO_OP, <comma separated list of columns>
    :param column_list: entire list of column names to drop when DROP_ALL action is specified
    :return: list of column names or empty list based on action chosen
    """

    if action.lower() == 'DROP_ALL'.lower():
        return column_list
    elif action.lower() != 'NO_OP'.lower():
        try:
            return [column.strip() for column in action.split(',')]
        except RuntimeError:
            logger.error(f"Invalid action : {action}. "
                         "Expected 'DROP_ALL' or 'NO_OP' or comma separated list of column names")
    return []


def identify_rows_with_majority_missing_values(df: DataFrame, threshold: float = 0.8) -> List[int]:
    """
    Identifies rows in the dataframe where most columns have missing values.
    Such data points aren't good for model learning and should be removed.

    :param df: pandas dataframe of the dataset
    :param threshold: fraction defining the minimum % of missing values in the row
    :return: list of row indices which have more missing values than the specified threshold
    """

    threshold_count = round((df.shape[1] - 1) * threshold)  # subtracting one to account for the target column
    logger.debug(f"Threshold count of missing row values : {threshold_count}")
    return [idx for idx in df.index if df.iloc[idx].isnull().sum() >= threshold_count]


def cleanup(df: DataFrame, auto=False):
    """
    Performs the following inplace cleanup operations on the dataframe:
    1. Remove duplicate rows
    2. Identify features with only one unique value
    3. Identify features which could be potential identifiers
    4. Identify features with majority missing values
    5. Identify rows with missing values

    Offers 2 modes of operation.
    1. Auto -> performs auto clean up. Default action is 'DROP_ALL' affected columns
    2. Suggestive -> takes user input on each feature cleanup operation

    :param df: pandas dataframe of the dataset
    :param auto: True for auto mode. False for suggestive mode. (Default : False)
    """

    logger.debug(f"Dataframe initial shape : {df.shape}")
    df.drop_duplicates(inplace=True)
    logger.debug(f"Dropped duplicate rows from dataframe. Dataframe shape : {df.shape}")

    single_value_columns = identify_single_value_columns(df)
    logger.debug(f"Identified single value features : {single_value_columns}")

    id_columns = identify_potential_id_columns(df)
    logger.debug(f"Identified id features : {id_columns}")

    missing_value_columns = identify_columns_with_majority_missing_values(df)
    logger.debug(f"Identified missing value features : {id_columns}")

    columns_to_drop = []

    if not auto:
        if single_value_columns:
            action = input(f"Features : {single_value_columns} have only one unique value. "
                           f"Actions -> DROP_ALL, NO_OP, <provide column names comma separated> ")
            columns_to_drop += __process_cleanup_action(action, single_value_columns)
        if id_columns:
            action = input(f"Features : {id_columns} are potential integer identifier columns. "
                           "Actions -> DROP_ALL, NO_OP, <provide column names comma separated> ")
            columns_to_drop += __process_cleanup_action(action, id_columns)
        if missing_value_columns:
            action = input(f"Features : {missing_value_columns} have majority missing values. "
                           "Actions -> DROP_ALL, NO_OP, <provide column names comma separated> ")
            columns_to_drop += __process_cleanup_action(action, missing_value_columns)
    else:
        columns_to_drop = single_value_columns + id_columns + missing_value_columns

    if columns_to_drop:
        columns_to_drop = set([column for column in columns_to_drop if column in df.columns])
        logger.debug(f"Dropping columns : {columns_to_drop} from the dataframe")

        df.drop(columns=columns_to_drop, inplace=True)
        logger.debug(f"Dropped {len(columns_to_drop)} columns from the dataframe. Dataframe shape : {df.shape}")
    else:
        logger.debug(f"No columns to drop. Dataframe shape : {df.shape}")

    missing_value_rows = identify_rows_with_majority_missing_values(df)
    if not auto:
        if missing_value_rows:
            action = input(f"Rows : {missing_value_rows} have majority missing values. "
                           "Actions -> DROP_ALL, NO_OP, <provide row indices comma separated> ")
            if action.lower() not in ('NO_OP'.lower(), 'DROP_ALL'.lower()):
                try:
                    missing_value_rows = filter(lambda x: x < df.shape[0],
                                                map(lambda x: int(x.strip()), action.split(',')))
                    logger.debug(f"Selected row indices : {missing_value_rows}")
                except RuntimeError:
                    logger.error(f"Invalid action : {action}. "
                                 "Expected 'DROP_ALL' or 'NO_OP' or comma separated list of row indices")
    if missing_value_rows:
        logger.debug(f"Dropping indices : {missing_value_rows} from the dataframe")
        df.drop(index=missing_value_rows, inplace=True)
        logger.debug(f"Dropped {len(missing_value_rows)} rows from the dataframe. Dataframe shape : {df.shape}")
    else:
        logger.debug(f"No rows to drop. Dataframe shape : {df.shape}")
