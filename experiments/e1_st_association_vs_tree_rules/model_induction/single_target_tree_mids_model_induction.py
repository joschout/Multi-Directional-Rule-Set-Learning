import os
from typing import List, Tuple, Dict

import pandas as pd
from dask import delayed
from dask.delayed import Delayed
from distributed import Client

from experiments.dask_utils.computations import compute_delayed_functions
from experiments.dask_utils.dask_initialization import reconnect_client_to_ssh_cluster
from experiments.utils.experiment_logging import create_logger, close_logger
from experiments.utils.file_creation import file_does_not_exist_or_has_been_created_earlier_than_

from experiments.arcbench_data_preparation.reworked_one_hot_encoding import get_original_data_fold_abs_file_name, \
    TrainTestEnum

# --- NAMING ---
# Tree rules
from experiments.e1_st_association_vs_tree_rules.file_naming.rules.single_target_filtered_tree_rule_naming import \
    get_single_target_tree_rules_absolute_file_name
from experiments.e1_st_association_vs_tree_rules.file_naming.single_target_filtered_mids_naming import (
    # Classifier names
    get_single_target_filtered_tree_mids_clf_abs_file_name,
    get_single_target_tree_mids_clf_relative_file_name,

    # Directory
    assoc_vs_tree_based_single_target_mids_clf_dir
)
from experiments.file_naming.single_target_classifier_indicator import SingleTargetClassifierIndicator
# ----------------

from mdrsl.data_structures.rules.multi_target_class_association_rule import MCAR
from mdrsl.rule_models.mids.io_mids import load_mcars, store_mids_classifier
from mdrsl.rule_models.mids.mids_classifier import MIDSClassifier
from mdrsl.rule_models.mids.model_fitting.mids_with_value_reuse import MIDSValueReuse

TargetAttr = str


def learn_single_target_tree_mids_model_for_dataset_fold_confidence(
        dataset_name: str,
        fold_i: int,
        target_attribute: str,
        confidence_boundary: float,
        min_support: float,
        max_depth: int
):
    classifier_indicator = SingleTargetClassifierIndicator.random_forest

    relative_name: str = get_single_target_tree_mids_clf_relative_file_name(
        dataset_name=dataset_name, fold_i=fold_i,
        target_attribute=target_attribute,
        classifier_indicator=classifier_indicator,
        min_support=min_support, max_depth=max_depth,
        confidence_boundary_val=confidence_boundary
    )
    log_file_dir: str = assoc_vs_tree_based_single_target_mids_clf_dir()
    logger = create_logger(
        logger_name=f'learn_single_target_tree_mids_' + relative_name,
        log_file_name=os.path.join(
            log_file_dir,
            f'{relative_name}_model_induction_single_target_tree_mids.log')
    )

    # --- load train data ---------------------------------------------------------------------------------------------
    # read in original (discretized) training data
    original_train_data_fold_abs_file_name = get_original_data_fold_abs_file_name(dataset_name, fold_i,
                                                                                  TrainTestEnum.train)
    df_train_original_column_order = pd.read_csv(original_train_data_fold_abs_file_name, delimiter=',')

    # --- load association rules --------------------------------------------------------------------------------------

    tree_clf_derived_rules_abs_file_name: str = get_single_target_tree_rules_absolute_file_name(
        dataset_name=dataset_name, fold_i=fold_i, target_attribute=target_attribute,
        classifier_indicator=classifier_indicator,
        min_support=min_support, max_depth=max_depth, confidence_boundary_val=confidence_boundary
    )

    logger.info(f"Reading MCARs from file: {tree_clf_derived_rules_abs_file_name}")
    print(tree_clf_derived_rules_abs_file_name)
    mcars: List[MCAR] = load_mcars(tree_clf_derived_rules_abs_file_name)
    logger.info(f"ground set size (nb of initial MCARs): {len(mcars)}")

    # --- Fit and save classifier -------------------------------------------------------------------------------------
    algorithm = "RDGS"
    debug_mids_fitting = False

    mids = MIDSValueReuse()
    mids.normalize = True
    logger.info("start MIDS model induction")
    mids.fit(df_train_original_column_order,
             use_targets_from_rule_set=True,
             class_association_rules=mcars, debug=debug_mids_fitting, algorithm=algorithm,
             )
    logger.info("finished MIDS model induction")
    mids_classifier: MIDSClassifier = mids.classifier
    logger.info(mids_classifier)

    logger.info("start saving MIDS model")
    tree_based_mids_classifier_abs_file_name = get_single_target_filtered_tree_mids_clf_abs_file_name(
        dataset_name=dataset_name, fold_i=fold_i,
        target_attribute=target_attribute,
        classifier_indicator=classifier_indicator,
        min_support=min_support, max_depth=max_depth,
        confidence_boundary_val=confidence_boundary
    )
    store_mids_classifier(tree_based_mids_classifier_abs_file_name, mids_classifier)
    logger.info(f"finished saving MIDS model to file: {tree_based_mids_classifier_abs_file_name}")
    close_logger(logger)


def main():
    from experiments.arcbench_data_preparation.dataset_info import datasets
    datasets = [dict(filename="iris", targetvariablename="class", numerical=True)]
    from experiments.dask_utils.dask_initialization import scheduler_host_name
    scheduler_host: str = scheduler_host_name
    list_of_computations: List[Tuple[Delayed, Dict]] = []

    classifier_indicator = SingleTargetClassifierIndicator.random_forest

    confidence_boundary_values: List[float] = [0.75, 0.95]
    min_support = 0.1
    max_depth = 7

    nb_of_folds: int = 10

    use_dask = False
    if use_dask:
        client: Client = reconnect_client_to_ssh_cluster(scheduler_host)

    for dataset_info in datasets:
        dataset_name = dataset_info['filename']
        target_attribute: str = dataset_info['targetvariablename']
        for fold_i in range(nb_of_folds):
            for confidence_boundary_val in confidence_boundary_values:

                n_days_in_hours = 24.0 * 0

                # do if file not exists or has been created more than 72 hours ago
                tree_based_mids_classifier_abs_file_name = get_single_target_filtered_tree_mids_clf_abs_file_name(
                    dataset_name=dataset_name, fold_i=fold_i,
                    target_attribute=target_attribute,
                    classifier_indicator=classifier_indicator,
                    min_support=min_support, max_depth=max_depth,
                    confidence_boundary_val=confidence_boundary_val
                )
                should_refit: bool = file_does_not_exist_or_has_been_created_earlier_than_(
                    tree_based_mids_classifier_abs_file_name,
                    n_days_in_hours
                )
                if should_refit:
                    if use_dask:
                        func_args = dict(
                            dataset_name=dataset_name, fold_i=fold_i, target_attribute=target_attribute,
                            confidence_boundary=confidence_boundary_val,
                            min_support=min_support,
                            max_depth=max_depth
                        )

                        delayed_func = \
                            delayed(
                                learn_single_target_tree_mids_model_for_dataset_fold_confidence)(
                                    **func_args
                            )
                        list_of_computations.append((delayed_func, func_args))
                    else:
                        learn_single_target_tree_mids_model_for_dataset_fold_confidence(
                            dataset_name=dataset_name, fold_i=fold_i, target_attribute=target_attribute,
                            confidence_boundary=confidence_boundary_val,
                            min_support=min_support,
                            max_depth=max_depth
                        )

    if use_dask:
        log_file_dir: str = assoc_vs_tree_based_single_target_mids_clf_dir()

        logger_name = f'learn_single_target_tree_mids_ERROR_LOGGER'
        logger_file_name = os.path.join(
            log_file_dir,
            f'ERROR_LOG_model_induction_single_target_tree_mids.log')

        compute_delayed_functions(
            list_of_computations=list_of_computations,
            client=client,
            nb_of_retries_if_erred=5,
            error_logger_name=logger_name,
            error_logger_file_name=logger_file_name
        )


if __name__ == '__main__':
    main()
