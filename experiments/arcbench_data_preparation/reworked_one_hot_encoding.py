import os
from enum import Enum
from logging import Logger
from typing import Optional
import pandas as pd

from mdrsl.data_handling.one_hot_encoding.encoding_book_keeping import EncodingBookKeeper
from mdrsl.data_handling.one_hot_encoding.encoding_io import store_encoding_book_keeper

from experiments.utils.experiment_logging import create_logger, close_logger
from experiments.file_naming.column_encodings import get_encodings_book_keeper_abs_file_name_for

from project_info import project_dir

original_data_dir_relative_root = 'data/arcBench/folds_discr2'
processed_data_dir_relative_root = 'data/arcBench_processed/'
one_hot_encoded_data_dir_relative_root = 'data/arcBench_processed/folds_discr2_one_hot_encoded'


class TrainTestEnum(Enum):
    train = 'train'
    test = 'test'


def get_original_fold_data_dir(train_test: Optional[TrainTestEnum] = None) -> str:
    if train_test is None:
        original_data_dir = os.path.join(project_dir, original_data_dir_relative_root)
    else:
        original_data_dir = os.path.join(project_dir, original_data_dir_relative_root, train_test.value)
    if not os.path.exists(original_data_dir):
        os.makedirs(original_data_dir)
    return original_data_dir


def get_original_data_fold_abs_file_name(dataset_name: str, fold_i: int, train_test: TrainTestEnum) -> str:
    original_data_dir: str = get_original_fold_data_dir(train_test)
    original_data_fold_abs_file_name: str = os.path.join(original_data_dir, f"{dataset_name}{str(fold_i)}.csv")
    return original_data_fold_abs_file_name


def get_original_full_data_abs_file_name(dataset_name: str, fold_i: int) -> str:
    original_full_data_dir: str = os.path.join(project_dir, processed_data_dir_relative_root, 'full_data')
    if not os.path.exists(original_full_data_dir):
        os.makedirs(original_full_data_dir)

    original_full_data_abs_file_name = os.path.join(original_full_data_dir, f'{dataset_name}{fold_i}.csv')
    return original_full_data_abs_file_name


# --- one-hot encoded -------------------------------------------

def get_one_hot_encoded_fold_data_dir(train_test: Optional[TrainTestEnum] = None) -> str:
    if train_test is None:
        one_hot_encoded_data_dir = os.path.join(project_dir, one_hot_encoded_data_dir_relative_root)
    else:
        one_hot_encoded_data_dir = os.path.join(project_dir, one_hot_encoded_data_dir_relative_root, train_test.value)
    if not os.path.exists(one_hot_encoded_data_dir):
        os.makedirs(one_hot_encoded_data_dir)
    return one_hot_encoded_data_dir


def get_one_hot_encoded_data_fold_abs_file_name(dataset_name: str, fold_i: int, train_test: TrainTestEnum) -> str:
    one_hot_encoded_data_dir: str = get_one_hot_encoded_fold_data_dir(train_test)
    one_hot_encoded_data_fold_abs_file_name: str = os.path.join(one_hot_encoded_data_dir,
                                                                f"{dataset_name}{str(fold_i)}.csv")
    return one_hot_encoded_data_fold_abs_file_name


def get_one_hot_encoded_full_data_abs_file_name(dataset_name: str, fold_i: int) -> str:
    one_hot_encoded_full_data_dir: str = os.path.join(project_dir, processed_data_dir_relative_root,
                                                      'full_data_one_hot_encoded')
    if not os.path.exists(one_hot_encoded_full_data_dir):
        os.makedirs(one_hot_encoded_full_data_dir)

    one_hot_encoded_full_data_abs_file_name = os.path.join(one_hot_encoded_full_data_dir, f'{dataset_name}{fold_i}.csv')
    return one_hot_encoded_full_data_abs_file_name


def convert_to_categorical(dataframe: pd.DataFrame, dataset_name: str, fold_i: int, logger: Optional[Logger]=None) -> pd.DataFrame:
    for column in dataframe.columns:
        column_type = dataframe[column].dtype
        if column_type != object:
            dataframe[column] = dataframe[column].astype('object')
            if logger is not None:
                logger.info(f"{dataset_name}{fold_i}: changed type of column {column} from {column_type} to object")
            else:
                print(f"{dataset_name}{fold_i}: changed type of column {column} from {column_type} to object")

    return dataframe


def one_hot_encode_dataset_fold(dataset_name: str, fold_i: int, ohe_prefix_separator: str) -> None:
    """
    One-hot encodes each of the Arch-bench fold train-test splits.

    """
    logger = create_logger(
        logger_name=f'one_hot_encode{dataset_name}{fold_i}',
        log_file_name=os.path.join(get_one_hot_encoded_fold_data_dir(train_test=None),
                                   f"{dataset_name}{fold_i}.log")
    )
    drop_first = False

    # === For fold i ====

    # --- Read in the original train and test data from archbench -----------------------------------------------------

    original_train_data_fold_abs_file_name = get_original_data_fold_abs_file_name(dataset_name, fold_i,
                                                                                  TrainTestEnum.train)
    original_test_data_fold_abs_file_name = get_original_data_fold_abs_file_name(dataset_name, fold_i,
                                                                                 TrainTestEnum.test)
    logger.info(f"Loading train fold: {original_train_data_fold_abs_file_name}")
    logger.info(f"Loading test fold: {original_test_data_fold_abs_file_name}")

    original_train_df = pd.read_csv(original_train_data_fold_abs_file_name, delimiter=',')
    original_test_df = pd.read_csv(original_test_data_fold_abs_file_name, delimiter=',')

    # --- Set each column to 'object' ------- -------------------------------------------------------------------------
    original_train_df = convert_to_categorical(original_train_df, dataset_name, fold_i)
    original_test_df = convert_to_categorical(original_test_df, dataset_name, fold_i)

    # --- Concatenate the train and test data for the current fold ----------------------------------------------------
    nb_of_train_examples = len(original_train_df)
    nb_of_test_examples = len(original_test_df)

    logger.info(f"Start concatenating train & test folds for {dataset_name}{fold_i}")
    original_concat_df = pd.concat([original_train_df, original_test_df], axis=0)
    if len(original_concat_df) != nb_of_train_examples + nb_of_test_examples:
        raise Exception("unexpected length")

    # --- Write out the full discretized dataset of this fold to file for inspection purposes -------------------------
    original_full_data_abs_file_name = get_original_full_data_abs_file_name(dataset_name, fold_i)
    logger.info(f"Writing out UN-encoded full dataset for {dataset_name}{fold_i}: {original_full_data_abs_file_name}")
    original_concat_df.to_csv(original_full_data_abs_file_name, index=False)

    # --- One-hot encoded the full data -------------------------------------------------------------------------------
    logger.info(f"Start one hot encoding {dataset_name}{fold_i}")
    one_hot_encoded_concat_df = pd.get_dummies(original_concat_df,
                                               prefix_sep=ohe_prefix_separator,
                                               drop_first=drop_first)

    one_hot_encoded_full_data_abs_file_name = get_one_hot_encoded_full_data_abs_file_name(dataset_name, fold_i)

    # --- Write out the one-hot encoded full data ---------------------------------------------------------------------
    logger.info(
        f"Writing out one hot encoded full dataset for {dataset_name}{fold_i}:"
        f" {one_hot_encoded_full_data_abs_file_name}")
    one_hot_encoded_concat_df.to_csv(one_hot_encoded_full_data_abs_file_name, index=False)

    # --- Create the EncodingBookKeeper and write it to file ----------------------------------------------------------
    encoding_book_keeper: EncodingBookKeeper = EncodingBookKeeper. \
        build_encoding_book_keeper_from_ohe_columns(one_hot_encoded_concat_df.columns,
                                                    ohe_prefix_separator=ohe_prefix_separator)
    logger.info(f"Creating one hot encoding book keeper for {dataset_name}{fold_i}")
    # %%
    encoding_book_keeper_abs_file_name = get_encodings_book_keeper_abs_file_name_for(dataset_name, fold_i)
    logger.info(f"Saving one hot encoding book keeper for {dataset_name}{fold_i}: {encoding_book_keeper_abs_file_name}")
    store_encoding_book_keeper(encoding_book_keeper_abs_file_name, encoding_book_keeper)

    # -- Split the full one-hot encoded dataset back into train and test ----------------------------------------------
    one_hot_encoded_train_df = one_hot_encoded_concat_df[:nb_of_train_examples]
    one_hot_encoded_test_df = one_hot_encoded_concat_df[nb_of_train_examples:]

    if len(one_hot_encoded_train_df) != nb_of_train_examples:
        raise Exception("unexpected length")
    if len(one_hot_encoded_test_df) != nb_of_test_examples:
        raise Exception("unexpected length")

    # -- Write out the one-hot encoded train and test -----------------------------------------------------------------
    one_hot_encoded_train_abs_file_name = get_one_hot_encoded_data_fold_abs_file_name(dataset_name, fold_i,
                                                                                      TrainTestEnum.train)
    one_hot_encoded_test_abs_file_name = get_one_hot_encoded_data_fold_abs_file_name(dataset_name, fold_i,
                                                                                     TrainTestEnum.test)
    logger.info(f"Saving one hot encoded train fold: {one_hot_encoded_train_abs_file_name}")
    logger.info(f"Saving one hot encoded test fold: {one_hot_encoded_test_abs_file_name}")

    one_hot_encoded_train_df.to_csv(one_hot_encoded_train_abs_file_name, index=False)
    one_hot_encoded_test_df.to_csv(one_hot_encoded_test_abs_file_name, index=False)
    logger.info("---")
    close_logger(logger)


def main():
    # from project_info import project_dir

    prefix_separator = ":=:"

    from experiments.arcbench_data_preparation.dataset_info import datasets
    # datasets = [dict(filename='labor')]
    nb_of_folds: int = 10

    for dataset_info in datasets:
        dataset_name = dataset_info['filename']

    # for dataset_name in ['australian', 'autos', 'credit-g', 'heart-statlog', 'ionosphere', 'segment', 'spambase']:
        for fold_i in range(nb_of_folds):
            one_hot_encode_dataset_fold(dataset_name, fold_i, ohe_prefix_separator=prefix_separator)


if __name__ == '__main__':
    main()
