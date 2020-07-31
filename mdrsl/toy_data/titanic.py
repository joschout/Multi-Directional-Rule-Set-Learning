import os
from typing import Tuple

import pandas as pd

from mdrsl.data_handling.split_train_test import train_test_split_pd


def get_total_df_titanic(data_dir: str) -> Tuple[pd.DataFrame, str]:
    df_det = pd.read_csv(os.path.join(data_dir, 'titanic_train.tab'),
                         ' ', header=None, names=['Passenger_Cat', 'Age_Cat', 'Gender'])
    df_y = pd.read_csv(os.path.join(data_dir, 'titanic_train.Y'), ' ', header=None, names=['Died', 'Survived'])
    df_total = df_det.join(df_y['Survived'])
    dataset_name = 'titanic'
    return df_total, dataset_name


def prepare_data_titanic(data_dir: str, prop=0.25) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    df_total, dataset_name = get_total_df_titanic(data_dir)

    # ---------------------------
    df_train, df_test = train_test_split_pd(df_total, prop=prop)

    return df_train, df_test, dataset_name
