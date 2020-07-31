import os
from typing import List, Tuple, Dict

import pandas as pd
from dask import delayed
from dask.delayed import Delayed

from dask_utils.computations import compute_delayed_functions
from dask_utils.dask_initialization import reconnect_client_to_ssh_cluster


from experiments.e1_st_association_vs_tree_rules.file_naming.rules.single_target_filtered_cars_naming import \
    assoc_vs_tree_based_single_target_car_dir, get_single_target_filtered_cars_abs_filename
from experiments.e1_st_association_vs_tree_rules.file_naming.rules.single_target_filtered_tree_rule_naming import \
    get_single_target_tree_rules_relative_file_name, get_single_target_tree_rules_absolute_file_name, \
    get_single_target_random_forest_absolute_file_name, get_single_target_tree_rules_gen_timing_info_absolute_file_name
from experiments.file_naming.column_encodings import get_encodings_book_keeper_abs_file_name_for
from experiments.file_naming.single_target_classifier_indicator import SingleTargetClassifierIndicator

from experiments.decision_tree_rule_learning.attribute_grouping import Attr
from experiments.decision_tree_rule_learning.data_preparation import PreparedDataForTargetSet
from experiments.e1_st_association_vs_tree_rules.rule_mining.tree_rule_mining.single_target_tree_ensemble_generation import \
    generate_n_single_target_tree_rules
from experiments.decision_tree_rule_learning.timing_utils import TreeRuleGenTimingInfo, store_tree_rule_gen_timing_info

from experiments.utils.experiment_logging import create_logger, close_logger


from mdrsl.data_handling.one_hot_encoding.encoding_io import load_encoding_book_keeper
from mdrsl.rule_models.mids.io_mids import load_mcars, store_mcars
from mdrsl.data_structures.rules.multi_target_class_association_rule import MCAR
from mdrsl.data_handling.one_hot_encoding.encoding_book_keeping import EncodingBookKeeper
from experiments.arcbench_data_preparation.reworked_one_hot_encoding import get_original_data_fold_abs_file_name, \
    TrainTestEnum, get_one_hot_encoded_data_fold_abs_file_name


def create_single_target_tree_based_mcars(
        dataset_name: str,
        fold_i: int,
        target_attribute: str,
        classifier_indicator: SingleTargetClassifierIndicator,
        confidence_boundary_val: float,
        min_support: float,
        max_depth: int,
        seed: int
):
    train_test = TrainTestEnum.train

    relative_name: str = get_single_target_tree_rules_relative_file_name(
        dataset_name=dataset_name, fold_i=fold_i,
        target_attribute=target_attribute,
        classifier_indicator=classifier_indicator,
        min_support=min_support, max_depth=max_depth,
        confidence_boundary_val=confidence_boundary_val
    )

    logger = create_logger(
        logger_name=f'create_single_target_tree_rules' + relative_name,
        log_file_name=os.path.join(assoc_vs_tree_based_single_target_car_dir(),
                                   f'{relative_name}_single_target_tree_rule_generation.log')
    )

    logger.info(f"Start reading MCARS for {dataset_name}{fold_i}_{target_attribute}"
                f" (confidence {confidence_boundary_val})")
    st_mcars_abs_file_name = get_single_target_filtered_cars_abs_filename(
        dataset_name, fold_i,
        target_attribute=target_attribute,
        confidence_boundary_val=confidence_boundary_val
    )

    filtered_st_mcars: List[MCAR] = load_mcars(st_mcars_abs_file_name)
    logger.info(f"Total nb of MCARS for {dataset_name}{fold_i}_{target_attribute}"
                f" (conf {confidence_boundary_val}): {len(filtered_st_mcars)}")

    n_tree_rules_to_generate = len(filtered_st_mcars)
    logger.info(f"Generate {n_tree_rules_to_generate} tree based rules")

    # --- load train data ---------------------------------------------------------------------------------------------
    df_original = pd.read_csv(get_original_data_fold_abs_file_name(dataset_name, fold_i, train_test),
                              delimiter=',')
    df_one_hot_encoded = pd.read_csv(get_one_hot_encoded_data_fold_abs_file_name(dataset_name, fold_i, train_test),
                                     delimiter=",")
    encoding_book_keeper: EncodingBookKeeper = load_encoding_book_keeper(
        get_encodings_book_keeper_abs_file_name_for(dataset_name, fold_i))

    # --- prepare data ------------------------------------------------------------------------------------------------

    original_group_to_predict: List[Attr] = [target_attribute]
    original_target_attr_set = set(original_group_to_predict)

    logger.info(f"Fetching the necessary columns for {dataset_name}{fold_i} {original_target_attr_set}")

    prepared_data: PreparedDataForTargetSet = PreparedDataForTargetSet.prepare_data_for_target_set(
        df_original=df_original,
        df_one_hot_encoded=df_one_hot_encoded,
        encoding_book_keeper=encoding_book_keeper,
        original_target_attr_set=original_target_attr_set,
    )

    random_forest_abs_file_name: str = get_single_target_random_forest_absolute_file_name(
        dataset_name=dataset_name, fold_i=fold_i,
        target_attribute=target_attribute,
        classifier_indicator=classifier_indicator,
        min_support=min_support, max_depth=max_depth,
        confidence_boundary_val=confidence_boundary_val
    )

    # --- Generate the required nb of tree-based rules ----------------------------------------------------------------
    logger.info(f"Start generating tree-based rules")
    tree_based_mcars: List[MCAR]
    tree_rule_gen_timing_info: TreeRuleGenTimingInfo
    tree_based_mcars, tree_rule_gen_timing_info = generate_n_single_target_tree_rules(
        n_tree_rules_to_generate=n_tree_rules_to_generate,
        prepared_data=prepared_data,
        encoding_book_keeper=encoding_book_keeper,
        min_support=min_support,
        max_depth=max_depth,
        logger=logger,
        seed=seed,
        random_forest_abs_file_name=random_forest_abs_file_name
    )

    # --- SAVE the generated tree-based rules
    tree_based_rules_abs_file_name: str = get_single_target_tree_rules_absolute_file_name(
        dataset_name=dataset_name, fold_i=fold_i,
        target_attribute=target_attribute,
        classifier_indicator=classifier_indicator,
        min_support=min_support, max_depth=max_depth,
        confidence_boundary_val=confidence_boundary_val
    )
    store_mcars(tree_based_rules_abs_file_name, tree_based_mcars)
    logger.info(f"finished writing tree-derived ruled to file: {tree_based_rules_abs_file_name}")

    tree_rule_gen_timing_info_abs_file_name: str = get_single_target_tree_rules_gen_timing_info_absolute_file_name(
        dataset_name=dataset_name, fold_i=fold_i,
        target_attribute=target_attribute,
        classifier_indicator=classifier_indicator,
        min_support=min_support, max_depth=max_depth,
        confidence_boundary_val=confidence_boundary_val
    )
    store_tree_rule_gen_timing_info(tree_rule_gen_timing_info_abs_file_name, tree_rule_gen_timing_info)

    logger.info("==================================================================")
    close_logger(logger)


def main():
    from experiments.arcbench_data_preparation.dataset_info import datasets
    datasets = [dict(filename="iris", targetvariablename="class", numerical=True)]
    from experiments.dask_utils.dask_initialization import scheduler_host_name
    scheduler_host: str = scheduler_host_name
    list_of_computations: List[Tuple[Delayed, Dict]] = []

    classifier_indicator: SingleTargetClassifierIndicator = SingleTargetClassifierIndicator.random_forest
    seed: int = 3
    min_support = 0.1
    max_depth = 7

    confidence_boundary_values: List[float] = [0.75, 0.95]

    use_dask = False
    if use_dask:
        client = reconnect_client_to_ssh_cluster(scheduler_host)

    for dataset_info in datasets:
        dataset_name = dataset_info['filename']
        target_attribute: str = dataset_info['targetvariablename']

        for fold_i in range(10):
            # --- Select subsets with a varying nb of rules (here: a varying confidence) --------------------------
            # --- For each of the different sets of association rules,
            #       generate a set of tree-based rules of the same size (*).
            for confidence_boundary_val in confidence_boundary_values:
                if use_dask:
                    func_args = dict(
                        dataset_name=dataset_name,
                        fold_i=fold_i,
                        target_attribute=target_attribute,
                        classifier_indicator=classifier_indicator,
                        confidence_boundary_val=confidence_boundary_val,
                        min_support=min_support,
                        max_depth=max_depth,
                        seed=seed
                    )

                    delayed_func = \
                        delayed(create_single_target_tree_based_mcars)(
                            **func_args
                        )
                    list_of_computations.append((delayed_func, func_args))
                else:
                    create_single_target_tree_based_mcars(
                        dataset_name=dataset_name,
                        fold_i=fold_i,
                        target_attribute=target_attribute,
                        classifier_indicator=classifier_indicator,
                        confidence_boundary_val=confidence_boundary_val,
                        min_support=min_support,
                        max_depth=max_depth,
                        seed=seed
                    )

            # Result: pairs of (association rule, tree-based rule) sets, with an increasing number of rules.

            # --- Learn an (M)IDS model for each of the two rule sets in a pair. ------------------------------

            # --- Evaluate the learned IDS models using the chosen evaluation metrics. ------------------------

            # --- Plot the evaluation metrics in function of the increasing number of rules. ------------------

    if use_dask:
        log_file_dir: str = assoc_vs_tree_based_single_target_car_dir()

        logger_name: str = 'create_single_target_tree_rules_ERROR_LOGGER'
        logger_file_name: str = os.path.join(
            log_file_dir,
            f'ERROR_LOG_single_target_tree_rule_generation.log'
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
