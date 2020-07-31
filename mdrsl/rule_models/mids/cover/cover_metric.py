import pandas as pd

from mdrsl.data_structures.rules.rule_part import Consequent
from mdrsl.rule_models.mids.cover.cover_checker import CoverChecker
from mdrsl.rule_models.mids.mids_rule import MIDSRule

import numpy as np


def get_avg_incorrect_cover_size(rule: MIDSRule, df: pd.DataFrame, cover_checker: CoverChecker,


                                 ) -> float:
    consequent: Consequent = rule.get_consequent()
    nb_of_attr_in_consequent: int = len(consequent)

    sum_of_incorrect_cover_sizes = 0

    cover = cover_checker.get_cover(rule, df)
    covers_size = np.sum(cover)
    # print(f"MIDS cover: {covers_size}")
    if not np.any(cover):
        raise Exception()

    for attr in consequent.get_attributes():
        incorrect_cover_size_for_attr: int = np.sum(cover_checker.get_incorrect_cover(rule, df, attr, cover=cover))
        sum_of_incorrect_cover_sizes += incorrect_cover_size_for_attr
    weighted_sum = sum_of_incorrect_cover_sizes / nb_of_attr_in_consequent
    # print(f"MIDS incorrect cover size: {weighted_sum} for rule {rule}")
    return weighted_sum
