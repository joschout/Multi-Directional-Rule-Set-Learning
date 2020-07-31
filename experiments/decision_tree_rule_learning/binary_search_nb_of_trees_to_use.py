import math
import time
from logging import Logger
from typing import Optional, List, Tuple

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from experiments.decision_tree_rule_learning.data_preparation import PreparedDataForTargetSet

from experiments.typing_utils import TimeDiffSec


def get_nb_of_leaf_nodes(tree_clf: DecisionTreeClassifier) -> int:
    children_left: np.ndarray = tree_clf.tree_.children_left
    children_right: np.ndarray = tree_clf.tree_.children_right

    n_leaves_of_tree_clf = np.count_nonzero(children_left == children_right)
    return n_leaves_of_tree_clf


def search_nb_of_multi_target_trees_to_use(
        n_tree_rules_to_generate: int,
        prepared_data_list: List[PreparedDataForTargetSet],
        min_support: float,
        max_depth: int,
        logger: Logger,
        seed: Optional[int] = None,

) -> Tuple[Optional[List[Tuple[PreparedDataForTargetSet, RandomForestClassifier]]], TimeDiffSec]:
    # nb_of_trees_to_use: int = 1
    nb_of_tree_based_rules_after_conversion: int = 0
    current_rf_list: Optional[List[Tuple[PreparedDataForTargetSet, RandomForestClassifier]]] = None
    total_time_random_forest_learning_s: TimeDiffSec = 0.0

    # min_nb_of_rfs = len(prepared_data_list)

    # --- estimate the nb of trees to use -------------------------------------------
    max_n_rules_in_tree = 2 ** max_depth
    min_n_trees_to_use = n_tree_rules_to_generate / max_n_rules_in_tree
    nb_of_rfs_to_use = len(prepared_data_list)
    estimate_nb_of_trees_per_rf: int = math.ceil(min_n_trees_to_use / nb_of_rfs_to_use)

    logger.info(f"INITIAL ESTIMATE: use {nb_of_rfs_to_use} RFs of each {estimate_nb_of_trees_per_rf} trees "
                f"for about {min_n_trees_to_use} trees in total")

    nb_of_trees_to_use = estimate_nb_of_trees_per_rf

    current_step_size = 1

    should_break = False
    while not should_break:
        nb_of_tree_based_rules_after_conversion = 0
        current_rf_list = []
        total_time_random_forest_learning_s = 0.0
        prepared_data: PreparedDataForTargetSet
        for prepared_data in prepared_data_list:

            start_time_decision_tree_learning_s = time.time()
            classifier: RandomForestClassifier = RandomForestClassifier(
                n_estimators=nb_of_trees_to_use,
                random_state=seed,
                min_samples_leaf=min_support,
                max_depth=max_depth
            )
            current_rf_clf = classifier

            # --- Learn a random forest given the current number of trees -----------------------------------
            classifier.fit(
                prepared_data.df_one_hot_encoded_descriptive_attributes,
                prepared_data.df_one_hot_encoded_target_attributes)
            end_time_decision_tree_learning_s = time.time()
            total_time_decision_tree_learning_s: float = end_time_decision_tree_learning_s - start_time_decision_tree_learning_s
            total_time_random_forest_learning_s += total_time_decision_tree_learning_s

            # --- b. Calculate its total number of leaves ----------------------------------
            tree_classifiers: List[DecisionTreeClassifier] = classifier.estimators_
            total_nb_of_leafs_in_random_forest: int = 0
            for tree_clf in tree_classifiers:
                total_nb_of_leafs_in_random_forest += get_nb_of_leaf_nodes(tree_clf)
            nb_of_tree_based_rules_after_conversion += total_nb_of_leafs_in_random_forest
            current_rf_list.append((prepared_data, current_rf_clf))

        if nb_of_tree_based_rules_after_conversion < n_tree_rules_to_generate:
            logger.info(f'Learned {len(current_rf_list)} RFs with each {nb_of_trees_to_use} trees'
                        f'--> {nb_of_tree_based_rules_after_conversion} rules '
                        f' < {n_tree_rules_to_generate} (goal)) '
                        f'--> INcreasing current step size {current_step_size} with 1')

            current_step_size += 1
            nb_of_trees_to_use += current_step_size
        if nb_of_tree_based_rules_after_conversion >= n_tree_rules_to_generate:
            should_break = True
        # else:
        #     logger.info(f'Learned {len(current_rf_list)} RFs with each {nb_of_trees_to_use} trees'
        #                 f'--> {nb_of_tree_based_rules_after_conversion} rules '
        #                 f' > {n_tree_rules_to_generate} (goal)) '
        #                 f'--> DEcreasing current step size {current_step_size} with 1')
        #     nb_of_trees_to_use -= current_step_size
        #     if current_step_size == 1:
        #         should_break = True
        #     current_step_size = 1
        #     nb_of_trees_to_use += 1

    logger.info(f'FINISHED search for tree rules: {len(current_rf_list)} RFs with each {nb_of_trees_to_use} trees'
                f'--> {nb_of_tree_based_rules_after_conversion} rules '
                f' > {n_tree_rules_to_generate} (goal)) ')

    return current_rf_list, total_time_random_forest_learning_s


def search_nb_of_single_target_trees_to_use(
        n_tree_rules_to_generate: int,
        prepared_data: PreparedDataForTargetSet,
        min_support: float,
        max_depth: int,
        logger: Logger,
        seed: Optional[int] = None,

) -> Tuple[Optional[RandomForestClassifier], TimeDiffSec]:

    nb_of_tree_based_rules_after_conversion: int = 0
    current_rf_clf: Optional[RandomForestClassifier] = None
    total_time_decision_tree_learning_s: TimeDiffSec = 0

    max_n_rules_in_tree: int = 2 ** max_depth
    min_n_trees_to_use = math.ceil(n_tree_rules_to_generate / max_n_rules_in_tree)
    nb_of_trees_to_use: int = min_n_trees_to_use

    current_step_size = 1

    should_break = False
    while not should_break:
        logger.info(f'Learning 1 RF using {nb_of_trees_to_use} trees...')
        nb_of_tree_based_rules_after_conversion = 0


        start_time_decision_tree_learning_s = time.time()
        current_rf_clf: RandomForestClassifier = RandomForestClassifier(
            n_estimators=nb_of_trees_to_use,
            random_state=seed,
            min_samples_leaf=min_support,
            max_depth=max_depth
        )

        # --- Learn a random forest given the current number of trees -----------------------------------
        current_rf_clf.fit(
            prepared_data.df_one_hot_encoded_descriptive_attributes,
            prepared_data.df_one_hot_encoded_target_attributes)
        end_time_decision_tree_learning_s = time.time()
        total_time_decision_tree_learning_s: TimeDiffSec = end_time_decision_tree_learning_s - start_time_decision_tree_learning_s

        # --- b. Calculate its total number of leaves ----------------------------------
        tree_classifiers: List[DecisionTreeClassifier] = current_rf_clf.estimators_
        total_nb_of_leafs_in_random_forest: int = 0
        for tree_clf in tree_classifiers:
            total_nb_of_leafs_in_random_forest += get_nb_of_leaf_nodes(tree_clf)
        nb_of_tree_based_rules_after_conversion += total_nb_of_leafs_in_random_forest

        if nb_of_tree_based_rules_after_conversion < n_tree_rules_to_generate:
            logger.info(f'Learned 1 RF with {nb_of_trees_to_use} trees'
                        f'--> {nb_of_tree_based_rules_after_conversion} rules '
                        f' < {n_tree_rules_to_generate} (goal)) '
                        f'--> INcreasing current step size {current_step_size} with 1')

            current_step_size += 1
            nb_of_trees_to_use += current_step_size
        if nb_of_tree_based_rules_after_conversion >= n_tree_rules_to_generate:
            should_break = True
        # else:
        #     logger.info(f'Learned {len(current_rf_list)} RFs with each {nb_of_trees_to_use} trees'
        #                 f'--> {nb_of_tree_based_rules_after_conversion} rules '
        #                 f' > {n_tree_rules_to_generate} (goal)) '
        #                 f'--> DEcreasing current step size {current_step_size} with 1')
        #     nb_of_trees_to_use -= current_step_size
        #     if current_step_size == 1:
        #         should_break = True
        #     current_step_size = 1
        #     nb_of_trees_to_use += 1

    logger.info(f'FINISHED search for tree rules: RF has {nb_of_trees_to_use} trees'
                f'--> {nb_of_tree_based_rules_after_conversion} rules '
                f' > {n_tree_rules_to_generate} (goal)) ')

    return current_rf_clf, total_time_decision_tree_learning_s
