import os
from typing import List, Tuple, Dict

from dask import delayed
from dask.delayed import Delayed
from dask.distributed import Client

from experiments.dask_utils.computations import compute_delayed_functions
from experiments.dask_utils.dask_initialization import reconnect_client_to_ssh_cluster
from experiments.arcbench_data_preparation.reworked_one_hot_encoding import get_original_data_fold_abs_file_name, \
    TrainTestEnum
from experiments.e2_multi_directional_model_comparison.file_naming.rules.single_target_tree_rule_naming import \
    get_single_target_tree_rule_dir
from experiments.e2_multi_directional_model_comparison.rule_mining.lib_single_target_tree_rule_generation import \
    learn_and_convert_single_target_tree_ensemble_to_rules
from experiments.utils.header_attributes import get_header_attributes


def main():
    from experiments.arcbench_data_preparation.dataset_info import datasets
    datasets = [dict(filename="iris", targetvariablename="class", numerical=True)]
    from experiments.dask_utils.dask_initialization import scheduler_host_name
    scheduler_host: str = scheduler_host_name
    min_support = 0.1
    max_depth = 7

    nb_of_trees_to_use_list: List[int] = [25, 50]

    list_of_computations: List[Tuple[Delayed, Dict]] = []
    use_dask = False
    if use_dask:
        client: Client = reconnect_client_to_ssh_cluster(scheduler_host)

    for dataset_info in datasets:
        dataset_name = dataset_info['filename']
        for fold_i in range(10):
            original_train_data_fold_abs_file_name = get_original_data_fold_abs_file_name(dataset_name, fold_i,
                                                                                          TrainTestEnum.train)

            target_columns: List[str] = get_header_attributes(original_train_data_fold_abs_file_name)
            for target_column in target_columns:
                target_attribute = str(target_column)

                for nb_of_trees_to_use in nb_of_trees_to_use_list:
                    if use_dask:
                        func_args = dict(
                            dataset_name=dataset_name, fold_i=fold_i, target_attribute=target_attribute,
                            nb_of_trees_per_model=nb_of_trees_to_use,
                            min_support=min_support, max_depth=max_depth
                        )

                        delayed_func = \
                            delayed(learn_and_convert_single_target_tree_ensemble_to_rules)(
                                **func_args
                            )
                        list_of_computations.append((delayed_func, func_args))
                    else:
                        learn_and_convert_single_target_tree_ensemble_to_rules(
                            dataset_name=dataset_name, fold_i=fold_i, target_attribute=target_attribute,
                            nb_of_trees_per_model=nb_of_trees_to_use,
                            min_support=min_support, max_depth=max_depth
                        )

            # Result: pairs of (association rule, tree-based rule) sets, with an increasing number of rules.

            # --- Learn an (M)IDS model for each of the two rule sets in a pair. ------------------------------

            # --- Evaluate the learned IDS models using the chosen evaluation metrics. ------------------------

            # --- Plot the evaluation metrics in function of the increasing number of rules. ------------------

    if use_dask:
        log_file_dir: str = get_single_target_tree_rule_dir()

        logger_name: str = f'mine_single_target_tree_mids_ERROR_LOGGER'
        logger_file_name: str = os.path.join(
            log_file_dir,
            f'ERROR_LOG_model_induction_single_target_tree_mids_easy.log'
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
