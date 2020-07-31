import math
from typing import Optional

import numpy as np

from mdrsl.data_structures.rules.multi_target_class_association_rule import MCAR
from mdrsl.data_structures.rules.rule_part import Antecedent, Consequent
from mdrsl.rule_models.mids.cover.cover_checker import CoverChecker
from mdrsl.rule_models.mids.mids_rule import MIDSRule


def add_support_and_confidence_to_MIDSRule(
        df_train, mids_rule: MIDSRule, cover_checker: Optional[CoverChecker]=None) -> None:
    add_support_and_confidence_to_MCAR(df_train=df_train, mcar=mids_rule.car, cover_checker=cover_checker)


def add_support_and_confidence_to_MCAR(df_train, mcar: MCAR, cover_checker: Optional[CoverChecker]=None) -> None:

    if cover_checker is None:
        cover_checker = CoverChecker()

    mids_rule: MIDSRule = MIDSRule(mcar)

    antecedent: Antecedent = mids_rule.get_antecedent()
    consequent: Consequent = mids_rule.get_consequent()

    ant_mask: np.ndarray = cover_checker._find_cover_mask_rule_part(antecedent, df_train)
    cons_mask: np.ndarray = cover_checker._find_cover_mask_rule_part(consequent, df_train)

    rule_mask: np.ndarray = np.logical_and(ant_mask, cons_mask)

    rule_support_count: int = np.sum(rule_mask)
    rule_support: float = rule_support_count / df_train.shape[0]

    # ---

    ant_support_count: int = np.sum(ant_mask)
    ant_support: float = ant_support_count / df_train.shape[0]
    confidence: float = rule_support / ant_support

    if math.isnan(confidence):
        confidence = 0

    mcar.confidence = float(confidence)
    mcar.support = float(rule_support)


