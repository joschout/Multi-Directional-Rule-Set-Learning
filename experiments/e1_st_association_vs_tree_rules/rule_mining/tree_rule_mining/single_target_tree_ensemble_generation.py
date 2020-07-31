"""
How to generate a set of tree-based rules of a given set size?
Note:
* A random forest will be converted to as many rules as it has leaves.
* Fitting 2 forests with the same settings will (with a high probability)
  generate a different number of leaves and thus a different number of rules.

Iterative algorithm (single-target case, multi-target is similar)
Given:
    A number of tree-based rules to generate.
Do:
    1. Choose at random a seed (to give to the random forest fitting procedure).
    2. Initialize the number of trees to use in a forest to 1
    3. While the number of generated tree-based rules is not larger than the number of association rules:
        a. Learn a random forest given the current number of trees
        b. Calculate its total number of leaves
        c. If the total number of leaves if larger than the number of tree-based rules to generate:
            Break
        d. Else: increment the number of rules to use per forest with 1
    4. Convert the random forest to tree-based rules.
    5. If the number of rules is strictly greater than the number of required rules:
        Sample the required number of rules
    6. Return the required number or rules

"""
import random
import time
from logging import Logger
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from mdrsl.data_structures.rules.multi_target_class_association_rule import MCAR
from mdrsl.data_handling.one_hot_encoding.encoding_book_keeping import EncodingBookKeeper
from mdrsl.rule_generation.decision_tree_conversion.support_confidence_adding import add_support_and_confidence_to_MIDSRule
from mdrsl.rule_generation.decision_tree_conversion.tree_to_rule_set_conversion import convert_decision_tree_to_mids_rule_list
from mdrsl.rule_models.mids.cover.cover_checker import CoverChecker
from mdrsl.rule_models.mids.mids_rule import MIDSRule

from experiments.decision_tree_rule_learning.attribute_grouping import Attr
from experiments.decision_tree_rule_learning.binary_search_nb_of_trees_to_use import \
    search_nb_of_single_target_trees_to_use
from experiments.decision_tree_rule_learning.timing_utils import TreeRuleGenTimingInfo
from experiments.decision_tree_rule_learning.data_preparation import PreparedDataForTargetSet
from experiments.file_naming.single_target_classifier_indicator import SingleTargetClassifierIndicator
from experiments.decision_tree_rule_learning.classifier_io import store_classifier

from experiments.typing_utils import TimeDiffSec


def get_nb_of_leaf_nodes(tree_clf: DecisionTreeClassifier) -> int:
    children_left: np.ndarray = tree_clf.tree_.children_left
    children_right: np.ndarray = tree_clf.tree_.children_right

    n_leaves_of_tree_clf = np.count_nonzero(children_left == children_right)
    return n_leaves_of_tree_clf


def convert_random_forest_to_rules(random_forest_clf: RandomForestClassifier,
                                   df_original_without_nans: pd.DataFrame,
                                   descriptive_one_hot_encoded_column_names: List[Attr],
                                   target_attribute_names: List[Attr],
                                   encoding_book_keeper: EncodingBookKeeper,
                                   logger: Logger
                                   ) -> Tuple[List[MCAR], TimeDiffSec]:
    tree_classifiers = random_forest_clf.estimators_

    total_time_rf_conversion_s = 0
    cover_checker = CoverChecker()
    rf_rule_list: List[MCAR] = []
    for tree_clf in tree_classifiers:
        list_of_dt_rules: Optional[List[MIDSRule]] = None
        try:
            start_time_clf_conversion_s = time.time()
            list_of_dt_rules: List[MIDSRule] = convert_decision_tree_to_mids_rule_list(
                tree_classifier=tree_clf,
                one_hot_encoded_feature_names=descriptive_one_hot_encoded_column_names,
                target_attribute_names=target_attribute_names,
                encoding_book_keeper=encoding_book_keeper)
        except NotImplementedError as err:
            logger.error(str(err))

        if list_of_dt_rules is not None:

            # --- adding support and confidence to rules
            mids_rule: MIDSRule
            for mids_rule in list_of_dt_rules:
                add_support_and_confidence_to_MIDSRule(df_original_without_nans, mids_rule, cover_checker=cover_checker)

            # logger.info(f"found {len(list_of_dt_rules)} rules,"
            #             f" updated total rule set size: {len(rf_rule_list)}")
            mids_rules_as_mcars = [mids_rule.car for mids_rule in list_of_dt_rules]
            rf_rule_list.extend(mids_rules_as_mcars)
            end_time_clf_conversion_s = time.time()
            total_time_clf_conversion_s = end_time_clf_conversion_s - start_time_clf_conversion_s
            total_time_rf_conversion_s += total_time_clf_conversion_s
    return rf_rule_list, total_time_rf_conversion_s


def _get_n_tree_based_rules_from(current_rf_list: List[Tuple[PreparedDataForTargetSet, RandomForestClassifier]],
                                 n_tree_rules_to_generate: int,
                                 encoding_book_keeper: EncodingBookKeeper,
                                 logger: Logger
                                 ) -> List[MCAR]:

    all_tree_based_rules: List[MCAR] = []
    for prepared_data, rf_clf in current_rf_list:
        # 4. Convert the random forest to tree-based rules.
        tree_based_rules, total_time_rf_conversion_s = convert_random_forest_to_rules(
            random_forest_clf=rf_clf,
            df_original_without_nans=prepared_data.df_original_without_nans_for_targets,
            descriptive_one_hot_encoded_column_names=prepared_data.descriptive_one_hot_encoded_columns,
            # target_attribute_names=df_original_target_attrs_without_nans.columns,
            target_attribute_names=prepared_data.target_one_hot_encoded_columns,
            encoding_book_keeper=encoding_book_keeper,
            logger=logger
        )
        all_tree_based_rules.extend(tree_based_rules)

    # 5. If the number of rules is strictly greater than the number of required rules:
    #     Sample the required number of rules
    if len(all_tree_based_rules) > n_tree_rules_to_generate:
        all_tree_based_rules = random.sample(all_tree_based_rules, n_tree_rules_to_generate)

    # writeout(tree_based_rules)  # should this be done here?
    # 6. Return the required number or rules
    return all_tree_based_rules


def generate_n_single_target_tree_rules(
        n_tree_rules_to_generate: int,
        prepared_data: PreparedDataForTargetSet,
        encoding_book_keeper: EncodingBookKeeper,
        min_support: float,
        max_depth: int,
        logger: Logger,
        random_forest_abs_file_name: str,
        seed: Optional[int] = None,
) -> Tuple[List[MCAR], TreeRuleGenTimingInfo]:
    if seed is None:
        raise Exception()
    if n_tree_rules_to_generate <= 0:
        raise Exception(f"n_tree_rules_to_generate = {n_tree_rules_to_generate} but should be larger than 0")

    logger.info(f'Start generating tree rules... Goal number: {n_tree_rules_to_generate}')

    # nb_of_trees_to_use: int = 1
    # nb_of_tree_based_rules_after_conversion: int = 0
    # current_rf_list: Optional[List[Tuple[PreparedDataForTargetSet, RandomForestClassifier]]] = None

    # prepared_data_list: List[PreparedDataForTargetSet] = []
    # for original_target_attribute_groups in attr_group_partitioning_list:
    #     attr_group: AttrGroup
    #     for attr_group in original_target_attribute_groups:
    #         prepared_data: PreparedDataForTargetSet = get_prepared_data_for_attr_group(
    #             original_group_to_predict=attr_group,
    #             df_original=df_original,
    #             df_one_hot_encoded=df_one_hot_encoded,
    #             encoding_book_keeper=encoding_book_keeper
    #         )
    #         prepared_data_list.append(prepared_data)

    optional_rf_classifier: Optional[RandomForestClassifier]
    total_time_decision_tree_learning_s: TimeDiffSec
    optional_rf_classifier, total_time_decision_tree_learning_s \
        = search_nb_of_single_target_trees_to_use(
            n_tree_rules_to_generate=n_tree_rules_to_generate,
            prepared_data=prepared_data,
            min_support=min_support,
            max_depth=max_depth,
            logger=logger,
            seed=seed
        )

    # -----------------------------------------------------------------------------------------------------------
    if optional_rf_classifier is None:
        raise Exception()
    else:
        logger.info(f'Learned RF has {len(optional_rf_classifier.estimators_)} trees')

        tree_based_rules: List[MCAR]
        total_time_rf_conversion_s: TimeDiffSec
        tree_based_rules, total_time_rf_conversion_s = convert_random_forest_to_rules(
            random_forest_clf=optional_rf_classifier,
            df_original_without_nans=prepared_data.df_original_without_nans_for_targets,
            descriptive_one_hot_encoded_column_names=prepared_data.descriptive_one_hot_encoded_columns,
            target_attribute_names=prepared_data.target_one_hot_encoded_columns,
            encoding_book_keeper=encoding_book_keeper,
            logger=logger
        )

        store_classifier(SingleTargetClassifierIndicator.random_forest, random_forest_abs_file_name,
                         optional_rf_classifier)
        logger.info(f"Wrote RF to {random_forest_abs_file_name}")

        if len(tree_based_rules) > n_tree_rules_to_generate:
            tree_based_rules = random.sample(tree_based_rules, n_tree_rules_to_generate)

        logger.info(f"REALITY: found {len(tree_based_rules)} tree based rules, wanted {n_tree_rules_to_generate}")
        for i in range(0, len(tree_based_rules)):
            logger.info(str(tree_based_rules[i]))
            if i > 10:
                break

        tree_rule_gen_timing_info = TreeRuleGenTimingInfo(
            total_time_decision_tree_learning_s=total_time_decision_tree_learning_s,
            total_time_rf_conversion_s=total_time_rf_conversion_s
        )

        return tree_based_rules, tree_rule_gen_timing_info
