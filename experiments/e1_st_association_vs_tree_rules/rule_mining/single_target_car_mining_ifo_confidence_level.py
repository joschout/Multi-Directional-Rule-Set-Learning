import os
from typing import List, Tuple, Dict

from dask import delayed
from dask.delayed import Delayed
from distributed import Client

from experiments.dask_utils.computations import compute_delayed_functions
from experiments.dask_utils.dask_initialization import reconnect_client_to_ssh_cluster
from experiments.utils.experiment_logging import create_logger, close_logger
from experiments.utils.header_attributes import get_header_attributes


from experiments.arcbench_data_preparation.arc_model_data_preparation import prepare_arc_data
from experiments.arcbench_data_preparation.reworked_one_hot_encoding import get_original_data_fold_abs_file_name, \
    TrainTestEnum

from experiments.e1_st_association_vs_tree_rules.file_naming.rules.single_target_filtered_cars_naming import (
    get_single_target_filtered_cars_abs_filename,
    get_single_target_filtered_cars_mining_timings_abs_filename,
    assoc_vs_tree_based_single_target_car_dir
)

from experiments.io_timings import store_timings_dict

from mdrsl.data_structures.rules.multi_target_class_association_rule import MCAR
from mdrsl.rule_models.mids.io_mids import store_mcars
from mdrsl.rule_generation.association_rule_mining.mlext_impl.mlext_interaction import mine_single_target_MCARs_mlext


def mine_cars_for_dataset_fold_target_attribute(
        dataset_name: str,
        fold_i: int,
        target_attribute: str,
        min_support: float,
        min_confidence: float,
        max_length: int,
):
    """
    1. load the required training data of the dataset fold.
    2. make sure the target attribute is the last attribute
    3. mine rules using the parameters settings
        --> check the number of rules!
    4. save the rules to file
    :return:
    """

    relative_name: str = f'{dataset_name}{fold_i}_{target_attribute}_{min_confidence}'

    logger = create_logger(
        logger_name=f'mine_filtered_single_target_cars_' + relative_name,
        log_file_name=os.path.join(assoc_vs_tree_based_single_target_car_dir(),
                                   f'{relative_name}_single_target_filtered_car_mining.log')
    )

    # logger.info(f"rule_cutoff={rule_cutoff}")

    # # load the required training data of the dataset fold.
    # original_train_data_fold_abs_file_name = get_original_data_fold_abs_file_name(
    #   dataset_name, fold_i, TrainTestEnum.train)
    # df_train_original_column_order = pd.read_csv(original_train_data_fold_abs_file_name, delimiter=',')
    # # 2. make sure the target attribute is the last attribute
    # df_train_reordered = reorder_columns(df_train_original_column_order, target_attribute)
    #
    # # REMOVE INSTANCES WITH NAN AS TARGET VALUE:
    # df_train_reordered = remove_instances_with_nans_in_column(df_train_reordered, target_attribute)
    df_train_reordered = prepare_arc_data(dataset_name, fold_i, target_attribute, TrainTestEnum.train)

    logger.info(f"start mining CARs for " + relative_name)

    st_mcars: List[MCAR]
    timings_dict: Dict[str, float]
    filtered_st_mcars, timings_dict = mine_single_target_MCARs_mlext(df_train_reordered,
                                                                     target_attribute=target_attribute,
                                                                     min_support=min_support,
                                                                     min_confidence=min_confidence,
                                                                     max_length=max_length)

    logger.info(f"finished mining CARs for {dataset_name} {fold_i}_{min_support}supp_{min_confidence}conf")
    logger.info(
        f"found {len(filtered_st_mcars)} CARs for {dataset_name} {fold_i}_{min_support}supp_{min_confidence}conf")

    filtered_st_mcars_abs_file_name: str = get_single_target_filtered_cars_abs_filename(
        dataset_name=dataset_name, fold_i=fold_i, target_attribute=target_attribute,
        confidence_boundary_val=min_confidence
    )
    store_mcars(filtered_st_mcars_abs_file_name, filtered_st_mcars)
    logger.info(f"finished writing CARs to file: {filtered_st_mcars_abs_file_name}")
    filtered_st_mcars_mining_timings_abs_file_name = get_single_target_filtered_cars_mining_timings_abs_filename(
        dataset_name=dataset_name, fold_i=fold_i, target_attribute=target_attribute,
        confidence_boundary_val=min_confidence
    )
    store_timings_dict(filtered_st_mcars_mining_timings_abs_file_name, timings_dict)

    close_logger(logger)


def main():
    from experiments.arcbench_data_preparation.dataset_info import datasets
    datasets = [dict(filename="iris", targetvariablename="class", numerical=True)]
    from experiments.dask_utils.dask_initialization import scheduler_host_name
    scheduler_host: str = scheduler_host_name
    list_of_computations: List[Tuple[Delayed, Dict]] = []

    min_support: float = 0.1
    max_length: int = 7
    confidence_boundary_values: List[float] = [0.75, 0.95]

    nb_of_folds: int = 10

    use_dask = False
    if use_dask:
        client: Client = reconnect_client_to_ssh_cluster(scheduler_host)

    for dataset_info in datasets:
        dataset_name = dataset_info['filename']

        for fold_i in range(nb_of_folds):
            original_train_data_fold_abs_file_name = get_original_data_fold_abs_file_name(dataset_name, fold_i,
                                                                                          TrainTestEnum.train)

            target_columns: List[str] = get_header_attributes(original_train_data_fold_abs_file_name)
            for target_column in target_columns:
                target_attribute = str(target_column)
                for conf_boundary_val in confidence_boundary_values:
                    if use_dask:
                        func_args = dict(
                            dataset_name=dataset_name,
                            fold_i=fold_i,
                            target_attribute=target_attribute,
                            min_support=min_support,
                            min_confidence=conf_boundary_val,
                            max_length=max_length
                        )
                        delayed_func = delayed(mine_cars_for_dataset_fold_target_attribute)(
                            **func_args
                        )
                        list_of_computations.append((delayed_func, func_args))
                    else:
                        mine_cars_for_dataset_fold_target_attribute(
                            dataset_name=dataset_name,
                            fold_i=fold_i,
                            target_attribute=target_attribute,
                            min_support=min_support,
                            min_confidence=conf_boundary_val,
                            max_length=max_length
                        )
        if use_dask:
            log_file_dir = assoc_vs_tree_based_single_target_car_dir()

            logger_name: str = 'mine_single_target_cars_ifo_confidence_bound_ERROR_LOGGER'
            logger_file_name: str = os.path.join(
                log_file_dir,
                f'ERROR_LOG_mine_single_target_cars_ifo_confidence_bound.log'
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
