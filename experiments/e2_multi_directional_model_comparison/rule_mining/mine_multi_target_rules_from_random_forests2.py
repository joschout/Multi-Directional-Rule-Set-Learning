import os
import time
from typing import List, Tuple, Dict

import pandas as pd
from dask import delayed
from dask.delayed import Delayed
from sklearn.ensemble import RandomForestClassifier

from experiments.dask_utils.computations import compute_delayed_functions
from experiments.dask_utils.dask_initialization import reconnect_client_to_ssh_cluster

from experiments.arcbench_data_preparation.reworked_one_hot_encoding import \
    get_one_hot_encoded_data_fold_abs_file_name
from experiments.arcbench_data_preparation.reworked_one_hot_encoding import get_original_data_fold_abs_file_name, \
    TrainTestEnum
from experiments.decision_tree_rule_learning.attribute_grouping import AttrGroupPartitioning, AttrGroup
from experiments.decision_tree_rule_learning.data_preparation import PreparedDataForTargetSet, get_attr_groupings, \
    get_prepared_data_for_attr_group
from experiments.decision_tree_rule_learning.tree_ensemble_generation import convert_random_forest_to_rules
from experiments.decision_tree_rule_learning.timing_utils import store_tree_rule_gen_timing_info, TreeRuleGenTimingInfo
from experiments.decision_tree_rule_learning.relative_file_naming import \
    get_tree_derived_rules_rel_file_name_without_extension

from experiments.utils.experiment_logging import create_logger, close_logger
from experiments.file_naming.car_naming import (
    get_tree_derived_rules_abs_file_name,
    get_tree_derived_rules_logger_abs_file_name, get_tree_derived_rules_dir,
    get_tree_derived_rules_gen_timing_info_abs_file_name
)
from experiments.file_naming.column_encodings import get_encodings_book_keeper_abs_file_name_for
from experiments.file_naming.single_target_classifier_indicator import SingleTargetClassifierIndicator

from mdrsl.data_structures.rules.multi_target_class_association_rule import MCAR
from mdrsl.data_handling.one_hot_encoding.encoding_book_keeping import EncodingBookKeeper
from mdrsl.data_handling.one_hot_encoding.encoding_io import load_encoding_book_keeper
from mdrsl.rule_models.mids.io_mids import store_mcars
from mdrsl.rule_models.mids.cover.cover_checker import CoverChecker

Attr = str


def learn_and_convert_tree_model_to_rules(
        dataset_name: str,
        fold_i: int,
        nb_of_trees_per_model: int,
        nb_of_original_targets_to_predict: int,
        nb_grouping_iterations: int,
        min_support: float,
        max_depth: int,
        seed: int
):
    classifier_indicator = SingleTargetClassifierIndicator.random_forest
    train_test = TrainTestEnum.train

    logger = create_logger(
        logger_name=f'mine_multi-target_cars_tree_derived_' + get_tree_derived_rules_rel_file_name_without_extension(
            dataset_name=dataset_name, fold_i=fold_i, classifier_indicator=classifier_indicator,
            nb_of_trees_per_model=nb_of_trees_per_model,
            nb_of_original_targets_to_predict=nb_of_original_targets_to_predict,
            min_support=min_support, max_depth=max_depth),
        log_file_name=get_tree_derived_rules_logger_abs_file_name(
            dataset_name=dataset_name, fold_i=fold_i,
            classifier_indicator=classifier_indicator, nb_of_trees_per_model=nb_of_trees_per_model,
            nb_of_original_targets_to_predict=nb_of_original_targets_to_predict,
            min_support=min_support, max_depth=max_depth
        )
    )

    # --- load train data ---------------------------------------------------------------------------------------------
    df_original = pd.read_csv(get_original_data_fold_abs_file_name(dataset_name, fold_i, train_test),
                              delimiter=',')
    df_one_hot_encoded = pd.read_csv(get_one_hot_encoded_data_fold_abs_file_name(dataset_name, fold_i, train_test),
                                     delimiter=",")

    encoding_book_keeper: EncodingBookKeeper = load_encoding_book_keeper(
        get_encodings_book_keeper_abs_file_name_for(dataset_name, fold_i))

    # --- prepare data ------------------------------------------------------------------------------------------------
    logger.info(f"Start preparing data using {nb_of_original_targets_to_predict} attrs per group"
                f" with {nb_grouping_iterations} grouping iterations")

    different_attr_groupings: List[AttrGroupPartitioning] = get_attr_groupings(
        nb_of_original_targets_to_predict=nb_of_original_targets_to_predict,
        nb_grouping_iterations=nb_grouping_iterations,
        encoding_book_keeper=encoding_book_keeper
    )

    complete_rule_list: List[MCAR] = []
    cover_checker = CoverChecker()

    total_time_random_forest_learning_s = 0.0
    total_time_rf_conversion_s = 0.0
    # prepared_data_list: List[PreparedDataForTargetSet] = []
    for original_target_attribute_partitioning in different_attr_groupings:
        attr_group: AttrGroup
        for attr_group in original_target_attribute_partitioning:
            prepared_data: PreparedDataForTargetSet = get_prepared_data_for_attr_group(
                original_group_to_predict=attr_group,
                df_original=df_original,
                df_one_hot_encoded=df_one_hot_encoded,
                encoding_book_keeper=encoding_book_keeper
            )
            # prepared_data_list.append(prepared_data)

            start_time_decision_tree_learning_s = time.time()
            classifier: RandomForestClassifier = RandomForestClassifier(
                n_estimators=nb_of_trees_per_model,
                random_state=seed,
                min_samples_leaf=min_support,
                max_depth=max_depth
            )

            # --- Learn a random forest given the current number of trees -----------------------------------
            classifier.fit(
                prepared_data.df_one_hot_encoded_descriptive_attributes,
                prepared_data.df_one_hot_encoded_target_attributes)
            end_time_decision_tree_learning_s = time.time()
            total_time_decision_tree_learning_s: float = end_time_decision_tree_learning_s - start_time_decision_tree_learning_s
            total_time_random_forest_learning_s += total_time_decision_tree_learning_s

            tree_based_rules: List[MCAR]
            total_time_rf_conversion_s: float
            tree_based_rules, partial_time_rf_conversion_s = convert_random_forest_to_rules(
                random_forest_clf=classifier,
                df_original_without_nans=prepared_data.df_original_without_nans_for_targets,
                descriptive_one_hot_encoded_column_names=prepared_data.descriptive_one_hot_encoded_columns,
                # target_attribute_names=df_original_target_attrs_without_nans.columns,
                target_attribute_names=prepared_data.target_one_hot_encoded_columns,
                encoding_book_keeper=encoding_book_keeper,
                logger=logger
            )
            total_time_rf_conversion_s += partial_time_rf_conversion_s
            complete_rule_list.extend(tree_based_rules)

    logger.info(f"Complete set size: {len(complete_rule_list)}")

    # --- Save rules to file ---------------------------------------------------------------------------------

    tree_clf_derived_rules_abs_file_name = get_tree_derived_rules_abs_file_name(dataset_name,
                                                                                fold_i,
                                                                                classifier_indicator,
                                                                                nb_of_trees_per_model,
                                                                                nb_of_original_targets_to_predict,
                                                                                min_support,
                                                                                max_depth)
    store_mcars(tree_clf_derived_rules_abs_file_name, complete_rule_list)
    logger.info(f"finished writing tree-derived ruled to file: {tree_clf_derived_rules_abs_file_name}")
    logger.info("==================================================================")

    tree_rule_gen_timing_info = TreeRuleGenTimingInfo(
        total_time_decision_tree_learning_s=total_time_random_forest_learning_s,
        total_time_rf_conversion_s=total_time_rf_conversion_s
    )

    tree_rule_gen_timing_info_abs_file_name: str = get_tree_derived_rules_gen_timing_info_abs_file_name(
        dataset_name,
        fold_i,
        classifier_indicator,
        nb_of_trees_per_model,
        nb_of_original_targets_to_predict,
        min_support,
        max_depth
    )
    store_tree_rule_gen_timing_info(tree_rule_gen_timing_info_abs_file_name, tree_rule_gen_timing_info)

    close_logger(logger)


def main():
    from experiments.arcbench_data_preparation.dataset_info import datasets
    datasets = [dict(filename="iris", targetvariablename="class", numerical=True)]
    from experiments.dask_utils.dask_initialization import scheduler_host_name
    scheduler_host: str = scheduler_host_name
    list_of_computations: List[Tuple[Delayed, Dict]] = []

    seed: int = 3
    nb_of_folds: int = 10
    nb_of_original_targets_to_predict: int = 2
    nb_grouping_iterations = 5

    nb_of_trees_per_model_list: List[int] = [5, 10]
    min_support: float = 0.1  # min_samples_leaf must be at least 1 or in (0, 0.5], got 0

    max_depth: int = 7 - nb_of_original_targets_to_predict

    use_dask = False
    if use_dask:
        client = reconnect_client_to_ssh_cluster(scheduler_host)

    for dataset_info in datasets:
        dataset_name = dataset_info['filename']

        for fold_i in range(nb_of_folds):

            for nb_of_trees_per_model in nb_of_trees_per_model_list:

                if use_dask:

                    func_args = dict(
                        dataset_name=dataset_name,
                        fold_i=fold_i,
                        nb_of_trees_per_model=nb_of_trees_per_model,
                        nb_of_original_targets_to_predict=nb_of_original_targets_to_predict,
                        nb_grouping_iterations=nb_grouping_iterations,
                        min_support=min_support,
                        max_depth=max_depth,
                        seed=seed
                    )

                    delayed_func = \
                        delayed(learn_and_convert_tree_model_to_rules)(
                            **func_args
                        )
                    list_of_computations.append((delayed_func, func_args))
                else:
                    learn_and_convert_tree_model_to_rules(
                        dataset_name=dataset_name,
                        fold_i=fold_i,
                        nb_of_trees_per_model=nb_of_trees_per_model,
                        nb_of_original_targets_to_predict=nb_of_original_targets_to_predict,
                        nb_grouping_iterations=nb_grouping_iterations,
                        min_support=min_support,
                        max_depth=max_depth,
                        seed=seed
                    )
    if use_dask:
        log_file_dir: str = get_tree_derived_rules_dir()

        logger_name: str = 'multi_target_tree_rule_generation_ERROR_LOGGER'
        logger_file_name: str = os.path.join(
            log_file_dir,
            f'ERROR_LOG_multi_target_tree_rule_generation.log'
        )

        compute_delayed_functions(
            list_of_computations=list_of_computations,
            client=client,
            nb_of_retries_if_erred=5,
            error_logger_name=logger_name,
            error_logger_file_name=logger_file_name
        )


if __name__ == '__main__':
    main()
