import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd

from mdrsl.data_structures.rules.rule_part import Consequent
from mdrsl.rule_models.mids.cover.rule_cover_cache import RuleCoverCache
from mdrsl.rule_models.mids.cover.cover_checker import CoverChecker, MIDSRule
from mdrsl.rule_models.mids.mids_ruleset import MIDSRuleSet


TargetAttr = str


class UncachedRuleException(Exception):
    pass


class CachedCoverChecker(CoverChecker):

    def __init__(self, rule_set: MIDSRuleSet, df: pd.DataFrame):
        self.rule_id_to_cover_cache = {}  # type: Dict[int, RuleCoverCache]

        for rule in rule_set.ruleset:  # type: MIDSRule
            self._prepare_cache_for_rule(rule, df)

    def get_cover(self, rule: MIDSRule, df: pd.DataFrame) -> np.ndarray:
        rule_id = rule.get_rule_id()
        if rule_id in self.rule_id_to_cover_cache:
            return self.rule_id_to_cover_cache[rule_id].cover
        else:
            raise UncachedRuleException("Uninitialized RuleCoverCache for rule: " + str(rule))

    def get_cover_size(self, rule: MIDSRule, df: pd.DataFrame) -> int:
        rule_id = rule.get_rule_id()
        if rule_id in self.rule_id_to_cover_cache:
            return self.rule_id_to_cover_cache[rule_id].cover_len
        else:
            raise UncachedRuleException("Uninitialized RuleCoverCache for rule: " + str(rule))

    def get_correct_cover(self,
                          rule: MIDSRule, df: pd.DataFrame,
                          target_attribute: TargetAttr,
                          cover: Optional[np.ndarray] = None) -> np.ndarray:
        rule_id = rule.get_rule_id()
        if rule_id in self.rule_id_to_cover_cache:
            return self.rule_id_to_cover_cache[rule_id].target_attr_to_correct_cover[target_attribute]
        else:
            raise UncachedRuleException("Uninitialized RuleCoverCache for rule: " + str(rule))

    def get_incorrect_cover(self,
                            rule: MIDSRule, df: pd.DataFrame,
                            target_attribute: TargetAttr,
                            cover: Optional[np.ndarray] = None
                            ) -> np.ndarray:
        rule_id = rule.get_rule_id()
        if rule_id in self.rule_id_to_cover_cache:
            return self.rule_id_to_cover_cache[rule_id].target_attr_to_incorrect_cover[target_attribute]
        else:
            raise UncachedRuleException("Uninitialized RuleCoverCache for rule: " + str(rule))

    def _prepare_cache_for_rule(self, rule: MIDSRule, df: pd.DataFrame):
        rule_id = rule.get_rule_id()
        if rule_id in self.rule_id_to_cover_cache:
            warnings.warn("rule_id already cached. Not caching it again")
            return

        cover: np.ndarray = super().get_cover(rule, df)
        cover_len: int = np.sum(cover)

        target_attr_to_correct_cover: Dict[str, np.ndarray] = {}
        target_attr_to_incorrect_cover: Dict[str, np.ndarray] = {}
        # rule_cover = {}

        target_attr_to_correct_cover_len: Dict[str, int] = {}
        target_attr_to_incorrect_cover_len: Dict[str, int] = {}

        # rule_cover_len = None

        consequent: Consequent = rule.get_consequent()
        target_attribute: TargetAttr
        for target_attribute in consequent.get_attributes():
            correct_cover_for_target_attr = super().get_correct_cover(
                rule, df, target_attribute, cover)
            target_attr_to_correct_cover[target_attribute] = correct_cover_for_target_attr

            incorrect_cover_for_target_attr = super().get_incorrect_cover(
                rule, df, target_attribute, cover)
            target_attr_to_incorrect_cover[target_attribute] = incorrect_cover_for_target_attr

            target_attr_to_correct_cover_len[target_attribute] = np.sum(target_attr_to_correct_cover)
            target_attr_to_incorrect_cover_len[target_attribute] = np.sum(target_attr_to_incorrect_cover)

        rule_cover_cache = RuleCoverCache(cover, cover_len,
                                          target_attr_to_correct_cover, target_attr_to_incorrect_cover,
                                          target_attr_to_incorrect_cover_len, target_attr_to_incorrect_cover_len)
        self.rule_id_to_cover_cache[rule_id] = rule_cover_cache
