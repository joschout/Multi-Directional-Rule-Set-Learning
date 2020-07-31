"""
A couple of problems can occur when fitting a model on the train data.
"""
from logging import Logger
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

TargetAttr = str


def assert_same_length(dataframe_descriptive_attrs: pd.DataFrame,
                       series_target_attr: pd.Series):
    nb_of_examples = dataframe_descriptive_attrs.shape[0]
    nb_of_labels = series_target_attr.size
    if nb_of_examples != nb_of_labels:
        raise Exception(f"The nb of train examples {nb_of_examples}"
                        f" does not match the nb of labels {nb_of_labels}")


def get_nan_mask(series: pd.Series) -> np.ndarray:
    nan_mask_series: pd.Series = series.isna()
    nan_mask_np: np.ndarray = nan_mask_series.values
    return nan_mask_np


def get_nan_mask_for_dataframe(dataframe: pd.DataFrame, targets_to_use: List[TargetAttr]) -> np.ndarray:
    nan_mask_dataframe: pd.DataFrame = dataframe.isna()

    nan_mask_np = np.ones(nan_mask_dataframe.shape[0], dtype=bool)
    # for column in nan_mask_dataframe.columns:
    for column in targets_to_use:
        nan_mask_np = np.logical_and(nan_mask_np, nan_mask_dataframe[column])
    return nan_mask_np


def remove_instances_with_nans_in_column(dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Fu

    Parameters
    ----------
    dataframe
    logger

    Returns
    -------

    """
    filtered_dataframe: pd.DataFrame = dataframe[dataframe[column_name].notna()]
    return filtered_dataframe


def filter_nans(dataframe_descriptive_attrs: pd.DataFrame,
                series_target_attr: pd.Series,
                logger: Optional[Logger]) -> Tuple[pd.DataFrame, pd.Series]:
    # -----------------------------------------------------------
    # DEALING WITH NANS
    nb_of_examples: int = dataframe_descriptive_attrs.shape[0]
    if logger:
        logger.info(f"\tNb of train instances {nb_of_examples}")

    nan_mask_np: np.ndarray = get_nan_mask(series_target_attr)
    not_nan_mask_np: np.ndarray = np.invert(nan_mask_np)

    nb_of_nans: int = np.count_nonzero(nan_mask_np)
    if logger:
        logger.info(f"\tNb of NaNs {nb_of_nans}/{nb_of_examples}")
        logger.info(f"\tFraction of NaNs: {nb_of_nans/nb_of_examples}")

    if nb_of_examples == nb_of_nans:
        raise Exception("\tTHERE ARE ONLY NaNs")
    elif nb_of_nans == 0:
        df_train_without_nans: pd.DataFrame = dataframe_descriptive_attrs
        series_target_attribute_without_nans = series_target_attr
    else:
        df_train_without_nans: pd.DataFrame = dataframe_descriptive_attrs[not_nan_mask_np]
        initial_type = series_target_attr.dtype
        series_target_attribute_without_nans = series_target_attr[not_nan_mask_np]
        series_target_attribute_without_nans = series_target_attribute_without_nans.infer_objects()
        inferred_type = series_target_attribute_without_nans.dtype
        if initial_type is not inferred_type and logger:
            logger.info(f"Changed target attr type from {initial_type} to {inferred_type}")
    return df_train_without_nans, series_target_attribute_without_nans


def filter_nans_multiple_target_atts(
        dataframe_descriptive_attrs: pd.DataFrame,
        dataframe_target_attrs: pd.DataFrame,
        logger: Optional[Logger]) -> Tuple[pd.DataFrame, pd.Series]:
    # -----------------------------------------------------------
    # DEALING WITH NANS
    nb_of_examples: int = dataframe_descriptive_attrs.shape[0]
    if logger:
        logger.info(f"\tNb of train instances {nb_of_examples}")

    nan_mask_np: np.ndarray = get_nan_mask_for_dataframe(dataframe_target_attrs, dataframe_target_attrs.columns)
    not_nan_mask_np: np.ndarray = np.invert(nan_mask_np)

    nb_of_rows_containing_nans: int = np.count_nonzero(nan_mask_np)
    if logger:
        logger.info(f"\tNb rows containing NaNs {nb_of_rows_containing_nans}/{nb_of_examples}")
        logger.info(f"\tFraction of rows containing NaNs: {nb_of_rows_containing_nans/nb_of_examples}")

    if nb_of_examples == nb_of_rows_containing_nans:
        raise Exception("\tTHERE ARE ONLY ROWS CONTAINING NaNs")
    elif nb_of_rows_containing_nans == 0:
        dataframe_train_without_nans: pd.DataFrame = dataframe_descriptive_attrs
        dataframe_target_attributes_without_nans = dataframe_target_attrs
    else:
        dataframe_train_without_nans: pd.DataFrame = dataframe_descriptive_attrs[not_nan_mask_np]

        initial_types = dataframe_target_attrs.dtypes
        dataframe_target_attributes_without_nans = dataframe_target_attrs[not_nan_mask_np]
        dataframe_target_attributes_without_nans = dataframe_target_attributes_without_nans.infer_objects()
        inferred_types = dataframe_target_attributes_without_nans.dtypes
        if initial_types is not inferred_types and logger:
            logger.info(f"Changed target attr type from {initial_types} to {inferred_types}")
    return dataframe_train_without_nans, dataframe_target_attributes_without_nans


