import random
from typing import Tuple

import pandas as pd


def train_test_split_pd(dataframe: pd.DataFrame, prop=0.25, seed=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if seed is not None:
        random.seed(seed)

    n = len(dataframe)
    samp = list(range(n))
    test_n = int(prop * n)
    train_n = n - test_n

    test_ind = random.sample(samp, test_n)
    train_ind = list(set(samp).difference(set(test_ind)))

    return dataframe.iloc[train_ind, :].reset_index(drop=True), dataframe.iloc[test_ind, :].reset_index(drop=True)
