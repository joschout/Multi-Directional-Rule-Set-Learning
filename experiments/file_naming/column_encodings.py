import os

from project_info import project_dir


def get_encodings_book_keeper_abs_file_name_for(dataset_name: str, fold_i: int) -> str:
    encodings_dir = os.path.join(project_dir, 'data/arcBench_processed/column_encodings')
    if not os.path.exists(encodings_dir):
        os.makedirs(encodings_dir)
    encoding_book_keeper_abs_file_name = os.path.join(
        encodings_dir, f'{dataset_name}{fold_i}.json.gz'
    )
    return encoding_book_keeper_abs_file_name
