from typing import List
from pandas import DataFrame
from itertools import chain

from dsbox.utils.logging import get_logger, LoggingLevel

__TRUE_VALUES_SET = {1, 'true', 'y', 'yes', }

__FALSE_VALUES_SET = {0, 'false', 'n', 'no', }

__BOOLEAN_VALUES_SET = __TRUE_VALUES_SET.union(__FALSE_VALUES_SET)

logger = get_logger('preprocessing', LoggingLevel.DEBUG)


def __is_boolean_value_column(unique_values: list) -> bool:
    """
    Checks if the unique values of a column can be mapped to boolean type
    1. At least one value should be mappable to True
    2. At least one value should be mappable to False
    3. Set of unique values (case-insensitive) should be a subset of __BOOLEAN_VALUES_SET (defined above)

    :param unique_values: list of unique values of a dataframe column
    :return True if column values can be mapped to boolean literals, False otherwise
    """
    has_true_value, has_false_value = False, False

    value_set = set()
    for value in unique_values:
        if isinstance(value, str):
            value = value.lower()  # for case insensitive comparisons
        value_set.add(value)

        if value in __TRUE_VALUES_SET:
            has_true_value = True
        elif value in __FALSE_VALUES_SET:
            has_false_value = True

    return has_true_value and has_false_value and value_set.issubset(__BOOLEAN_VALUES_SET)


def identify_potential_boolean_columns(df: DataFrame) -> List[str]:
    return [col for col in df.columns if __is_boolean_value_column(df[col].unique().tolist())]


def __create_boolean_mapping_dict(df: DataFrame, boolean_columns: List[str]) -> dict:
    """
    Maps unique values of the boolean columns to True / False

    :param df: pandas dataframe of the dataset
    :param boolean_columns: list of column names which are identified as boolean columns
    :return: mapping dict which is used to replace the current values with True / False
    """

    boolean_mapping_dict = {}
    unique_values = set(chain(*[df[col].unique().tolist() for col in boolean_columns]))
    logger.debug(f"Unique values for boolean columns : {unique_values}")

    for val in unique_values:
        v = val
        if isinstance(val, str):
            v = val.lower()

        if v in __TRUE_VALUES_SET:
            boolean_mapping_dict[val] = True
        elif v in __FALSE_VALUES_SET:
            boolean_mapping_dict[val] = False

    logger.debug(f"Boolean mapping dict for columns : {boolean_mapping_dict}")
    return boolean_mapping_dict


def update_columns_values_to_boolean(df: DataFrame, boolean_columns: List[str]):
    """
    Updates the column values to booleans -> True / False and explicitly changes column data type to bool

    :param df: pandas dataframe of the dataset
    :param boolean_columns: list of column names which are identified as boolean columns
    """

    mapping_dict = __create_boolean_mapping_dict(df, boolean_columns)
    df.replace(mapping_dict, inplace=True)
    logger.debug(f"Boolean column values replaced using mapping dict")
    for col in boolean_columns:
        df[col] = df[col].astype('bool')
    logger.debug(f"Updated datatype of boolean columns : {boolean_columns} to bool")


def preprocess(df: DataFrame, auto=False):
    """
    Performs the following inplace operations on the dataframe:
    1. Replace values from a potential boolean column (having values like 1, 0, "Y", "N", etc)
    to boolean literals -> True, False

    Offers 2 modes of operation.
    1. Auto -> performs auto preprocessing
    2. Suggestive -> takes user input on each operation

    :param df: pandas dataframe of the dataset
    :param auto: True for auto mode. False for suggestive mode. (Default : False)
    """

    boolean_columns = identify_potential_boolean_columns(df)
    logger.debug(f"Identified boolean value features : {boolean_columns}")

    if not auto:
        if boolean_columns:
            action = input(f"Features : {boolean_columns} have only 2 unique values which match the boolean pattern. "
                           f"Actions -> UPDATE_ALL, NO_OP, <provide column names comma separated> ")
            if action.lower() not in ('NO_OP'.lower(), 'UPDATE_ALL'.lower()):
                try:
                    boolean_columns = [col.strip() for col in action.split(',') if col in df.columns]
                    logger.debug(f"Selected boolean columns : {boolean_columns}")
                except RuntimeError:
                    logger.error(f"Invalid action : {action}. "
                                 "Expected 'UPDATE_ALL' or 'NO_OP' or comma separated list of column names")
    if boolean_columns:
        update_columns_values_to_boolean(df, boolean_columns)
