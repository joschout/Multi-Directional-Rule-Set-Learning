import warnings
from typing import Dict, Tuple, List, KeysView

import numpy as np
import pandas as pd

from mdrsl.rule_models.mids.cover.cover_checker import CoverChecker
from mdrsl.rule_models.mids.mids_rule import MIDSRule
from mdrsl.rule_models.mids.mids_ruleset import MIDSRuleSet

TargetAttr = str


class OverlapChecker:
    def __init__(self, cover_checker: CoverChecker, debug=True, ):
        self.cover_checker = cover_checker  # type: CoverChecker
        self.debug = debug

    def find_overlap_mask(self, rule1: MIDSRule, rule2: MIDSRule, df: pd.DataFrame) -> np.ndarray:
        """
        Compute the number of points which are covered both by r1 and r2 w.r.t. data frame df
        :param rule1:
        :param rule2:
        :param df:
        :return:
        """
        # if not rule1.head_contains_target_attribute(target_attribute) or not rule2.head_contains_target_attribute(
        #         target_attribute):
        #     raise Exception("WARNING: overlap check where one of the two rules does not contain target attribute",
        #                     str(target_attribute))

        # if rule1 == rule2:
        #     raise Exception("Dont check a rule with itself for overlap")
        return np.logical_and(
            self.cover_checker.get_cover(rule1, df),
            self.cover_checker.get_cover(rule2, df)
        )

    def get_relative_overlap_count(self, rule1: MIDSRule, rule2: MIDSRule,
                                   df: pd.DataFrame, target_attribute: TargetAttr) -> int:
        """
        Compute the overlap of rules RELATIVE to a certain target attribute.
        IF the target attribute isn't in both rule heads, the relative overlap is 0
        ELSE (the target attribute is in both heads):
            the relative overlap is equal to the normal overlap

        NOTE: the relative overlap is the same for ALL SHARED TARGET ATTRIBUTES of the two rules.
        For all other attributes, the relative overlap is 0.
        """
        if not rule1.head_contains_target_attribute(target_attribute) or not rule2.head_contains_target_attribute(
                target_attribute):
            return 0

        return self.get_pure_overlap_count(rule1, rule2, df)

    def get_pure_overlap_count(self, rule1: MIDSRule, rule2: MIDSRule, df: pd.DataFrame) -> int:
        """
        Find the overlap of two rules, independent of their heads.
        """
        overlap_mask = self.find_overlap_mask(rule1, rule2, df)  # type: np.ndarray
        overlap_count = np.sum(overlap_mask)  # type: int
        return overlap_count


class CachedOverlapChecker(OverlapChecker):

    def __init__(self, rule_set: MIDSRuleSet, df: pd.DataFrame, cover_checker: CoverChecker, debug=True):
        super().__init__(cover_checker, debug)
        self.pure_overlap_cache = {}  # type: Dict[Tuple[int,int], int]
        self.debug = debug

        rule_list = list(rule_set.ruleset)  # type: List[MIDSRule]
        nb_of_rules = len(rule_list)

        for i in range(0, nb_of_rules):
            for j in range(i + 1, nb_of_rules):
                rule_i = rule_list[i]
                rule_j = rule_list[j]

                # if there are any target variables in common, calculate the pure overlap
                target_attr_rule_i: KeysView[str] = rule_i.get_target_attributes()
                target_attr_rule_j: KeysView[str] = rule_j.get_target_attributes()
                if not target_attr_rule_i.isdisjoint(target_attr_rule_j):
                    cache_key = self.__get_cache_key(rule_i, rule_j)
                    self.pure_overlap_cache[cache_key] = super().get_pure_overlap_count(rule_i, rule_j, df)  # type: int

        print("overlap cache prepared")

    @staticmethod
    def __get_cache_key(rule1: MIDSRule, rule2: MIDSRule):
        rule1_id = rule1.get_rule_id()
        rule2_id = rule2.get_rule_id()
        if rule1_id < rule2_id:
            cache_key = (rule1_id, rule2_id)
        elif rule2_id < rule1_id:
            cache_key = (rule2_id, rule1_id)
        else:
            raise Exception("Dont check a rule with itself for overlap")
        return cache_key

    def get_relative_overlap_count(self, rule1: MIDSRule, rule2: MIDSRule, df: pd.DataFrame, target_attribute: str) -> int:
        """
        Compute the overlap of rules RELATIVE to a certain target attribute.
        IF the target attribute isn't in both rule heads, the relative overlap is 0
        ELSE (the target attribute is in both heads):
            the relative overlap is equal to the normal overlap

        NOTE: the relative overlap is the same for ALL SHARED TARGET ATTRIBUTES of the two rules.
        For all other attributes, the relative overlap is 0.
        """
        if not rule1.head_contains_target_attribute(target_attribute) or not rule2.head_contains_target_attribute(
                target_attribute):
            return 0

        overlap_mask = self.find_overlap_mask(rule1, rule2, df)  # type: np.ndarray
        overlap_count = np.sum(overlap_mask)
        return overlap_count

    def get_pure_overlap_count(self, rule1: MIDSRule, rule2: MIDSRule, df: pd.DataFrame) -> int:
        """
        Find the overlap of tho rules, independent of their heads.
        """
        cache_key = self.__get_cache_key(rule1, rule2)
        pure_overlap_count = self.pure_overlap_cache.get(cache_key, None)
        if pure_overlap_count is None:
            warnings.warn("Note: requested pure overlap of two rules with no shared target attributes."
                          " This is not cached and had to be calculated on the spot.")
            return super().get_pure_overlap_count(rule1, rule2, df)
        else:
            return pure_overlap_count

    # def calculate_overlap(self, all_rules, quant_dataframe):
    #     for rule in all_rules.ruleset:
    #         rule.prepare_cover_cache(quant_dataframe)
    #     print("cover cache prepared")
    #
    #     len_all_rules = len(all_rules)
    #     progress_bars = 20
    #     progress_bar_step = len_all_rules / progress_bars
    #     progress_bar_curr = 1
    #
    #     for i, rule_i in enumerate(all_rules.ruleset):
    #         for j, rule_j in enumerate(all_rules.ruleset):
    #             overlap_tmp = rule_i.rule_overlap(rule_j, quant_dataframe)
    #             overlap_len = np.sum(overlap_tmp)
    #
    #             self.overlap_cache[repr(rule_i) + repr(rule_j)] = overlap_len
    #
    #     print("overlap cache prepared")
