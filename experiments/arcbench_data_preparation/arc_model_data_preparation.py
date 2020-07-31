import pandas as pd

from experiments.arcbench_data_preparation.reworked_one_hot_encoding import get_original_data_fold_abs_file_name, \
    TrainTestEnum
from mdrsl.data_handling.nan_data_filtering import remove_instances_with_nans_in_column
from mdrsl.data_handling.reorder_dataset_columns import reorder_columns


def prepare_arc_data(
        dataset_name: str,
        fold_i: int,
        target_attribute: str,
        train_test: TrainTestEnum
) -> pd.DataFrame:
    # read in original (discretized) training/test data
    # reorder the data so the target column is last
    original_data_fold_abs_file_name = get_original_data_fold_abs_file_name(dataset_name, fold_i, train_test)
    df_original_column_order = pd.read_csv(original_data_fold_abs_file_name, delimiter=',')
    df_reordered = reorder_columns(df_original_column_order, target_attribute)

    # REMOVE INSTANCES WITH NAN AS TARGET VALUE:
    df_reordered = remove_instances_with_nans_in_column(df_reordered, target_attribute)
    return df_reordered
