from typing import List, Set

import numpy as np
import pandas as pd

from mdrsl.data_handling.one_hot_encoding.encoding_book_keeping import EncodingBookKeeper
from mdrsl.data_handling.nan_data_filtering import filter_nans_multiple_target_atts, \
    get_nan_mask_for_dataframe

from experiments.decision_tree_rule_learning.attribute_grouping import Attr, AttrGroupPartitioning, AttrGroup
from experiments.decision_tree_rule_learning.multiple_targets_per_decision_tree import \
    get_ohe_descriptive_and_target_attributes, get_original_target_attribute_partitioning


class PreparedDataForTargetSet:
    def __init__(self,
                 original_target_attr_set: Set[Attr],
                 df_original_without_nans_for_targets: pd.DataFrame,
                 df_one_hot_encoded_descriptive_attributes: pd.DataFrame,
                 df_one_hot_encoded_target_attributes: pd.DataFrame,

                 descriptive_one_hot_encoded_columns: List[Attr],
                 target_one_hot_encoded_columns: List[Attr]

                 ):
        self.original_target_attr_set: Set[Attr] = original_target_attr_set

        self.df_original_without_nans_for_targets: pd.DataFrame = df_original_without_nans_for_targets

        self.df_one_hot_encoded_descriptive_attributes: pd.DataFrame = df_one_hot_encoded_descriptive_attributes
        self.df_one_hot_encoded_target_attributes: pd.DataFrame = df_one_hot_encoded_target_attributes

        self.descriptive_one_hot_encoded_columns: List[Attr] = descriptive_one_hot_encoded_columns
        self.target_one_hot_encoded_columns: List[Attr] = target_one_hot_encoded_columns

    @staticmethod
    def prepare_data_for_target_set(
            df_original: pd.DataFrame,
            df_one_hot_encoded: pd.DataFrame,
            encoding_book_keeper: EncodingBookKeeper,

            original_target_attr_set: Set[Attr]
    ) -> 'PreparedDataForTargetSet':
        # --- Filter the data based on NaNs in the original target attributes -----------------------------------------
        df_original_target_attributes = df_original[original_target_attr_set]

        df_ohe_without_nans, df_original_target_attrs_without_nans = filter_nans_multiple_target_atts(
            df_one_hot_encoded, df_original_target_attributes, logger=None
        )

        nan_mask_np: np.ndarray = get_nan_mask_for_dataframe(df_original_target_attributes,
                                                             df_original_target_attributes.columns)
        not_nan_mask_np: np.ndarray = np.invert(nan_mask_np)
        df_original_without_nans_for_targets = df_original[not_nan_mask_np]

        # --- Find the ohe descriptive and target attributes ----------------------------------------------------------
        descriptive_one_hot_encoded_columns: List[Attr]
        target_one_hot_encoded_columns: List[Attr]
        descriptive_one_hot_encoded_columns, target_one_hot_encoded_columns = get_ohe_descriptive_and_target_attributes(
            original_target_attr_set, encoding_book_keeper)

        df_one_hot_encoded_descriptive_attributes: pd.DataFrame = df_ohe_without_nans[
            descriptive_one_hot_encoded_columns]

        df_one_hot_encoded_target_attributes: pd.DataFrame = df_ohe_without_nans[
            target_one_hot_encoded_columns]

        return PreparedDataForTargetSet(
            original_target_attr_set=original_target_attr_set,
            df_original_without_nans_for_targets=df_original_without_nans_for_targets,
            df_one_hot_encoded_descriptive_attributes=df_one_hot_encoded_descriptive_attributes,
            df_one_hot_encoded_target_attributes=df_one_hot_encoded_target_attributes,
            descriptive_one_hot_encoded_columns=descriptive_one_hot_encoded_columns,
            target_one_hot_encoded_columns=target_one_hot_encoded_columns
        )


def get_attr_groupings(
        nb_of_original_targets_to_predict: int,
        nb_grouping_iterations: int,
        encoding_book_keeper: EncodingBookKeeper
) -> List[AttrGroupPartitioning]:
    different_attr_groupings: List[AttrGroupPartitioning] = []

    for grouping_i in range(nb_grouping_iterations):
        original_target_attribute_groups: AttrGroupPartitioning = get_original_target_attribute_partitioning(
            encoding_book_keeper, nb_of_original_targets_to_predict)
        different_attr_groupings.append(original_target_attribute_groups)
    return different_attr_groupings


def get_prepared_data_for_attr_group(
        original_group_to_predict: AttrGroup,
        df_original: pd.DataFrame,
        df_one_hot_encoded: pd.DataFrame,
        encoding_book_keeper: EncodingBookKeeper,
) -> PreparedDataForTargetSet:
    original_target_attr_set = set(original_group_to_predict)
    prepared_data = PreparedDataForTargetSet.prepare_data_for_target_set(
        df_original=df_original,
        df_one_hot_encoded=df_one_hot_encoded,
        encoding_book_keeper=encoding_book_keeper,
        original_target_attr_set=original_target_attr_set,
    )
    return prepared_data
