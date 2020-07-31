from typing import Dict

import numpy as np


class RuleCoverCache:
    # originally:
    #  MIDSCacher caches overlap
    #  a Dict in ids.mids_rule.MIDSRule caches cover
    #
    def __init__(self,
                 cover: np.ndarray,
                 cover_len: int,

                 target_attr_to_correct_cover: Dict[str, np.ndarray],
                 target_attr_to_incorrect_cover: Dict[str, np.ndarray],

                 target_attr_to_correct_cover_len: Dict[str, int],
                 target_attr_to_incorrect_cover_len: Dict[str, int]):
        self.cover = cover  # type: np.ndarray

        self.target_attr_to_correct_cover = target_attr_to_correct_cover  # type: Dict[str, np.ndarray]
        self.target_attr_to_incorrect_cover = target_attr_to_incorrect_cover  # type: Dict[str, np.ndarray]
        # self.rule_cover = {}

        self.cover_len = cover_len  # type: int
        self.target_attr_to_correct_cover_len = target_attr_to_correct_cover_len  # type: Dict[str, int]
        self.target_attr_to_incorrect_cover_len = target_attr_to_incorrect_cover_len   # type: Dict[str, int]

        # # self.rule_cover_len = None
        #
        # cons = rule.car.consequent
        # for target_attribute in cons.itemset.keys():
        #     self.correct_cover[target_attribute] \
        #         = uncached_cover_checker.get_correct_cover(rule, df, target_attribute, self.cover)
        #     self.incorrect_cover[target_attribute] \
        #         = uncached_cover_checker.get_incorrect_cover(rule, df, target_attribute)
        #
        #     # self.cover_len = np.sum(self.cover)
        #     # self.correct_cover_len = np.sum(self.correct_cover)
        #     # self.incorrect_cover_len = np.sum(self.incorrect_cover)
        #     self.cover_len = len(self.cover)
        #     self.correct_cover_len = len(self.correct_cover)
        #     self.incorrect_cover_len = len(self.incorrect_cover)
        #     # self.rule_cover_len = np.sum(self.cover_cache.rule_cover)
        #
        # self.cache_prepared = True
