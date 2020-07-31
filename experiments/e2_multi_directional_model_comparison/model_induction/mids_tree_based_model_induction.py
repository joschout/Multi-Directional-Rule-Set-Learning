import os
from typing import List, Tuple, Dict

import pandas as pd
from dask import delayed
from dask.delayed import Delayed

from experiments.dask_utils.computations import compute_delayed_functions
from experiments.dask_utils.dask_initialization import reconnect_client_to_ssh_cluster

from experiments.arcbench_data_preparation.reworked_one_hot_encoding import get_original_data_fold_abs_file_name, \
    TrainTestEnum
from experiments.utils.experiment_logging import create_logger, close_logger
from experiments.file_naming.car_naming import get_tree_derived_rules_abs_file_name
from experiments.file_naming.classifier_naming import get_tree_based_mids_clf_abs_file_name, \
    get_tree_based_mids_dir
from experiments.file_naming.single_target_classifier_indicator import SingleTargetClassifierIndicator

from mdrsl.data_structures.rules.multi_target_class_association_rule import MCAR
from mdrsl.rule_models.mids.io_mids import load_mcars, store_mids_classifier
from mdrsl.rule_models.mids.mids_classifier import MIDSClassifier
from mdrsl.rule_models.mids.model_fitting.mids_with_value_reuse import MIDSValueReuse

TargetAttr = str


def learn_tree_based_mids_model_for_dataset_fold(
        dataset_name: str,
        fold_i: int,
        classifier_indicator: SingleTargetClassifierIndicator,
        nb_of_trees_per_model: int,
        nb_of_original_targets_to_predict: int,
        min_support: float,
        max_depth: int
):

    logger = create_logger(
        logger_name=f'learn_mids_model{dataset_name}{fold_i}_tree_derived_rules',
        log_file_name=os.path.join(get_tree_based_mids_dir(),
                                   f'{dataset_name}{fold_i}_model_induction_tree_derived_rules.log')
    )
    # --- load train data ---------------------------------------------------------------------------------------------
    # read in original (discretized) training data
    df_original_train = pd.read_csv(get_original_data_fold_abs_file_name(dataset_name, fold_i, TrainTestEnum.train),
                                    delimiter=',')

    # --- load association rules --------------------------------------------------------------------------------------
    tree_clf_derived_rules_abs_file_name = get_tree_derived_rules_abs_file_name(dataset_name,
                                                                                fold_i,
                                                                                classifier_indicator,
                                                                                nb_of_trees_per_model,
                                                                                nb_of_original_targets_to_predict,
                                                                                min_support,
                                                                                max_depth)
    logger.info(f"Reading MCARs from file: {tree_clf_derived_rules_abs_file_name}")
    mcars: List[MCAR] = load_mcars(tree_clf_derived_rules_abs_file_name)
    logger.info(f"ground set size (nb of initial MCARs): {len(mcars)}")

    # --- Fit and save classifier -------------------------------------------------------------------------------------
    algorithm = "RDGS"
    debug_mids_fitting = False

    mids = MIDSValueReuse()
    mids.normalize = True
    logger.info("start MIDS model induction")

    mids.fit(df_original_train,
             class_association_rules=mcars, debug=debug_mids_fitting, algorithm=algorithm,
             # lambda_array=lambda_array
             use_targets_from_rule_set=False,
             )
    logger.info("finished MIDS model induction")
    mids_classifier: MIDSClassifier = mids.classifier
    logger.info(mids_classifier)
    logger.info(f"Selected {len(mids_classifier.rules)} out of {len(mcars)} rules "
                f"({(len(mids_classifier.rules) / len(mcars) *100):.2f}%)")

    logger.info("start saving MIDS model")
    tree_based_mids_classifier_abs_file_name = get_tree_based_mids_clf_abs_file_name(
        dataset_name=dataset_name, fold_i=fold_i,
        classifier_indicator=classifier_indicator, nb_of_trees_per_model=nb_of_trees_per_model,
        nb_of_original_targets_to_predict=nb_of_original_targets_to_predict,
        min_support=min_support, max_depth=max_depth
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

    nb_of_folds: int = 10
    classifier_indicator = SingleTargetClassifierIndicator.random_forest
    nb_of_original_targets_to_predict: int = 2

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
                        classifier_indicator=classifier_indicator,
                        nb_of_trees_per_model=nb_of_trees_per_model,
                        nb_of_original_targets_to_predict=nb_of_original_targets_to_predict,
                        min_support=min_support,
                        max_depth=max_depth)

                    delayed_func = \
                        delayed(learn_tree_based_mids_model_for_dataset_fold)(
                            **func_args
                        )
                    list_of_computations.append((delayed_func, func_args))
                else:
                    learn_tree_based_mids_model_for_dataset_fold(
                        dataset_name=dataset_name,
                        fold_i=fold_i,
                        classifier_indicator=classifier_indicator,
                        nb_of_trees_per_model=nb_of_trees_per_model,
                        nb_of_original_targets_to_predict=nb_of_original_targets_to_predict,
                        min_support=min_support,
                        max_depth=max_depth
                    )

    if use_dask:
        log_file_dir: str = get_tree_based_mids_dir()

        logger_name: str = 'model_induction_tree_derived_rules_ERROR_LOGGER'
        logger_file_name: str = os.path.join(
            log_file_dir,
            f'ERROR_LOG_model_induction_tree_derived_rules.log'
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
