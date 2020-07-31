import os
from typing import List, Dict, Tuple

import pandas as pd
from dask import delayed
from dask.delayed import Delayed
from distributed import Client

from dask_utils.computations import compute_delayed_functions
from dask_utils.dask_initialization import reconnect_client_to_ssh_cluster


from experiments.arcbench_data_preparation.reworked_one_hot_encoding import get_original_data_fold_abs_file_name, \
    TrainTestEnum

# --- NAMING ---
from experiments.e2_multi_directional_model_comparison.file_naming.merged_single_target_mids_naming import (
    get_merged_single_target_tree_mids_clf_abs_file_name,
    get_merged_single_target_tree_mids_relative_file_name_without_extension,
    get_merged_single_target_mids_clf_dir
)

from experiments.e2_multi_directional_model_comparison.file_naming.single_target_mids_naming import \
    get_single_target_tree_mids_clf_abs_file_name
from experiments.e2_multi_directional_model_comparison.file_naming.rules.single_target_tree_rule_naming import (
    get_single_target_tree_rules_gen_timing_info_abs_file_name)
# -----------------

from experiments.decision_tree_rule_learning.timing_utils import TreeRuleGenTimingInfo, load_tree_rule_gen_timing_info

from experiments.utils.experiment_logging import create_logger, close_logger
from experiments.utils.header_attributes import get_header_attributes
from experiments.file_naming.single_target_classifier_indicator import SingleTargetClassifierIndicator

from mdrsl.rule_models.mids.io_mids import load_mids_classifier
from mdrsl.rule_models.mids.mids_classifier import MIDSClassifier
from mdrsl.rule_models.mids.model_fitting.mids_with_value_reuse import MIDSValueReuse
from mdrsl.rule_models.eids.merged_model_io import store_merged_st_mids_model
from mdrsl.rule_models.eids.st_to_mt_model_merging import MergedSTMIDSClassifier

TargetAttr = str


def merge_single_target_mids_models_for_dataset_fold(
        dataset_name: str,
        fold_i: int,
        nb_of_trees_to_use: int,
        min_support: float,
        max_depth: int
):
    classifier_indicator = SingleTargetClassifierIndicator.random_forest

    relative_name: str = get_merged_single_target_tree_mids_relative_file_name_without_extension(
        dataset_name=dataset_name, fold_i=fold_i,
        classifier_indicator=classifier_indicator,
        nb_of_trees_per_model=nb_of_trees_to_use,
        min_support=min_support, max_depth=max_depth,
    )
    log_file_dir: str = get_merged_single_target_mids_clf_dir()

    logger_name: str = f'merge_single_target_mids_models__' + relative_name
    logger_file_name: str = os.path.join(
        log_file_dir,
        f'{relative_name}_model_merging_single_target_tree_mids.log'
    )

    logger = create_logger(
        logger_name=logger_name,
        log_file_name=logger_file_name
    )

    original_train_data_fold_abs_file_name = get_original_data_fold_abs_file_name(dataset_name, fold_i,
                                                                                  TrainTestEnum.train)

    target_columns: List[str] = get_header_attributes(original_train_data_fold_abs_file_name)

    merged_st_clf = MergedSTMIDSClassifier()

    for target_attribute in target_columns:
        st_mids_classifier_abs_file_name: str = get_single_target_tree_mids_clf_abs_file_name(
            dataset_name=dataset_name, fold_i=fold_i,
            target_attribute=target_attribute,
            classifier_indicator=classifier_indicator,
            nb_of_trees_per_model=nb_of_trees_to_use,
            min_support=min_support, max_depth=max_depth
        )

        # --- load single target classifier ---------------------------------------------------------------------------
        logger.info(f"start loading MIDS model from {st_mids_classifier_abs_file_name}")
        st_mids_classifier: MIDSClassifier = load_mids_classifier(st_mids_classifier_abs_file_name)
        logger.info("finished loading MIDS model")
        logger.info(st_mids_classifier)
        reconstructed_mids = MIDSValueReuse()
        reconstructed_mids.classifier = st_mids_classifier
        merged_st_clf.add_single_target_model(st_mids_classifier)

        st_tree_rule_gen_timing_info_abs_file_name: str = get_single_target_tree_rules_gen_timing_info_abs_file_name(
            dataset_name, fold_i, target_attribute,
            classifier_indicator, nb_of_trees_to_use, min_support, max_depth
        )
        st_tree_rule_gen_timing_info: TreeRuleGenTimingInfo = load_tree_rule_gen_timing_info(
            st_tree_rule_gen_timing_info_abs_file_name)

        st_total_time_decision_tree_learning_s = st_tree_rule_gen_timing_info.total_time_decision_tree_learning_s
        st_total_time_rf_conversion_s = st_tree_rule_gen_timing_info.total_time_rf_conversion_s

        st_total_rule_gen_time_s: float = st_total_time_decision_tree_learning_s + st_total_time_rf_conversion_s
        merged_st_clf.add_rule_generation_time(st_total_rule_gen_time_s)

    # --- load test data ----------------------------------------------------------------------------------------------
    # read in original (discretized) training data
    original_test_data_fold_abs_file_name = get_original_data_fold_abs_file_name(dataset_name, fold_i,
                                                                                 TrainTestEnum.test)
    df_test_original_column_order = pd.read_csv(original_test_data_fold_abs_file_name,
                                                delimiter=',')

    merged_st_clf.calculate_ruleset_interpretability_statistics(
        test_dataframe=df_test_original_column_order, target_attributes=target_columns)

    # --- Evaluate and store predictive performance  ------------------------------------------------------------------
    filter_nans: bool = True
    merged_st_clf.calculate_score_info(test_dataframe=df_test_original_column_order, filter_nans=filter_nans)
    logger.info("Evaluated MERGED MIDS classifier on predictive performance")

    # --- Evaluate and store interpretability statistics --------------------------------------------------------------
    merged_st_clf.calculate_ruleset_interpretability_statistics(
        test_dataframe=df_test_original_column_order, target_attributes=target_columns)
    logger.info("Evaluated MIDS classifier on interpretability")

    # --- store merged classifier ------------------------------------------------------------------------------------
    logger.info("start saving merged single target MIDS model")
    merged_st_clf_abs_file_name: str = get_merged_single_target_tree_mids_clf_abs_file_name(
        dataset_name=dataset_name,
        fold_i=fold_i,
        classifier_indicator=classifier_indicator,
        nb_of_trees_per_model=nb_of_trees_to_use,
        min_support=min_support, max_depth=max_depth
    )
    store_merged_st_mids_model(merged_st_clf_abs_file_name, merged_st_clf)
    logger.info(f"finished saving merged single target MIDS model to file: {merged_st_clf_abs_file_name}")
    logger.info("---")

    close_logger(logger)


def main():
    from experiments.arcbench_data_preparation.dataset_info import datasets
    datasets = [dict(filename="iris", targetvariablename="class", numerical=True)]
    from experiments.dask_utils.dask_initialization import scheduler_host_name
    scheduler_host: str = scheduler_host_name
    list_of_computations: List[Tuple[Delayed, Dict]] = []

    min_support = 0.1
    max_depth = 7
    nb_of_trees_to_use_list: List[int] = [50]

    nb_of_folds: int = 10

    use_dask = False
    if use_dask:
        client: Client = reconnect_client_to_ssh_cluster(scheduler_host)

    for dataset_info in datasets:
        dataset_name = dataset_info['filename']
        for fold_i in range(nb_of_folds):
            for nb_of_trees_to_use in nb_of_trees_to_use_list:
                if use_dask:
                    func_args = dict(
                        dataset_name=dataset_name, fold_i=fold_i,
                        nb_of_trees_to_use=nb_of_trees_to_use, min_support=min_support,
                        max_depth=max_depth
                    )

                    delayed_func = \
                        delayed(merge_single_target_mids_models_for_dataset_fold)(
                            **func_args
                        )
                    list_of_computations.append((delayed_func, func_args))
                else:
                    merge_single_target_mids_models_for_dataset_fold(
                        dataset_name=dataset_name, fold_i=fold_i,
                        nb_of_trees_to_use=nb_of_trees_to_use, min_support=min_support,
                        max_depth=max_depth
                    )

    if use_dask:
        log_file_dir: str = get_merged_single_target_mids_clf_dir()

        logger_name: str = f'merge_single_target_tree_mids_ERROR_LOGGER'
        logger_file_name: str = os.path.join(
            log_file_dir,
            f'ERROR_LOG_model_merging_single_target_tree_mids.log'
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
