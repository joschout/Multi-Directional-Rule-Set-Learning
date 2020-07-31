import os

from project_info import project_dir

number_encoded_dataset_dir = os.path.join(project_dir, 'data/arcBench_processed/number_encoded_datasets')


def get_item_encoder_abs_file_name_for(dataset_name: str, fold_i: int) -> str:
    item_encoder_dir = number_encoded_dataset_dir
    if not os.path.exists(item_encoder_dir):
        os.makedirs(item_encoder_dir)
    item_encoder_abs_file_name = os.path.join(
        item_encoder_dir, f'{dataset_name}{fold_i}_item_encoder.pickle'
    )
    return item_encoder_abs_file_name


def get_encoded_transactions_abs_file_name_for(dataset_name: str, fold_i: int) -> str:
    encoded_transactions_dir = number_encoded_dataset_dir
    if not os.path.exists(encoded_transactions_dir):
        os.makedirs(encoded_transactions_dir)
    encoded_transactions_abs_file_name = os.path.join(
        encoded_transactions_dir, f'{dataset_name}{fold_i}_encoded_transactions.txt'
    )
    return encoded_transactions_abs_file_name


def get_encoded_frequent_itemsets_abs_file_name_for(dataset_name: str, fold_i: int) -> str:
    encoded_frequent_itemsets_dir = number_encoded_dataset_dir
    if not os.path.exists(encoded_frequent_itemsets_dir):
        os.makedirs(encoded_frequent_itemsets_dir)
    encoded_frequent_itemsets_abs_file_name = os.path.join(
        encoded_frequent_itemsets_dir, f'{dataset_name}{fold_i}_encoded_frequent_itemsets.txt'
    )
    return encoded_frequent_itemsets_abs_file_name


def get_lcm_command() -> str:
    return os.path.join(project_dir, 'external/lcm53/lcm')
