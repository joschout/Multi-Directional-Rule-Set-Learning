# See also:
#     http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/
#     http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/
#     http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns/#fpgrowth

import time

import pandas as pd
from typing import Tuple, Dict

from typing import List, Optional

from pyarc.data_structures.car import ClassAssocationRule
from pyarc.data_structures.antecedent import Antecedent as CARAntecedent
from pyarc.data_structures.consequent import Consequent as CARConsequent
from pyarc.data_structures.item import Item as CARItem

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules, class_based_association_rules
from mlxtend.frequent_patterns.association_rules import class_based_association_rules_for_all_targets

from mdrsl.data_structures.rules.multi_target_class_association_rule import MCAR
from mdrsl.data_structures.rules.generalized_rule_part import GeneralizedAntecedent
from mdrsl.data_structures.item import EQLiteral
from mdrsl.data_structures.rules.rule_part import Consequent as MCARConsequent

from mdrsl.rule_generation.association_rule_mining.frequent_itemset_mining import (
    attribute_value_separator, dataframe_to_list_of_transactions)

antecedents_column: str = 'antecedents'
consequents_column: str = 'consequents'
support_column: str = 'support'
confidence_column: str = 'confidence'


def mine_single_target_CARs_mlext(df_train: pd.DataFrame,
                                  target_attribute: str,
                                  min_support: float = 0.1, min_confidence: float = 0.5,
                                  max_length=None
                                  ) -> List[ClassAssocationRule]:
    transactions: List[List[str]] = dataframe_to_list_of_transactions(df_train)
    df_frequent_itemsets: pd.DataFrame = mine_frequent_itemsets_mlext_fpgrowth(
        transactions=transactions, min_support=min_support, max_len=max_length
    )
    df_class_based_assoc_rules: pd.DataFrame = generate_class_based_association_rules_from_frequent_itemsets(
        df_frequent_itemsets=df_frequent_itemsets, target_attribute=target_attribute,
        min_confidence=min_confidence
    )
    single_target_cars = convert_mlext_association_rules_dataframe_to_single_target_CARs(
        df_association_rules=df_class_based_assoc_rules)
    return single_target_cars


def mine_single_target_MCARs_mlext(
        df_train: pd.DataFrame,
        target_attribute: str,
        min_support: float = 0.1, min_confidence: float = 0.5,
        max_length=None
) -> Tuple[List[MCAR], Dict[str, float]]:
    transactions: List[List[str]] = dataframe_to_list_of_transactions(df_train)

    start_fim_time_s = time.time()
    df_frequent_itemsets: pd.DataFrame = mine_frequent_itemsets_mlext_fpgrowth(
        transactions=transactions, min_support=min_support, max_len=max_length
    )
    end_fim_time_s = time.time()

    start_assoc_time = end_fim_time_s
    df_class_based_assoc_rules: pd.DataFrame = generate_class_based_association_rules_from_frequent_itemsets(
        df_frequent_itemsets=df_frequent_itemsets, target_attribute=target_attribute,
        min_confidence=min_confidence
    )
    single_target_cars = convert_mlext_association_rules_dataframe_to_MCARs(
        df_association_rules=df_class_based_assoc_rules)

    end_assoc_time = time.time()
    total_fim_time_s = end_fim_time_s - start_fim_time_s
    total_assoc_time_s = end_assoc_time - start_assoc_time

    timing_info = dict(
        total_fim_time_s=total_fim_time_s,
        total_assoc_time_s=total_assoc_time_s
    )
    return single_target_cars, timing_info


def mine_single_target_CARs_mlext_for_all_targets(
        df_train: pd.DataFrame,
        min_support: float = 0.1, min_confidence: float = 0.5,
        max_length=None
) -> List[ClassAssocationRule]:
    transactions: List[List[str]] = dataframe_to_list_of_transactions(df_train)
    df_frequent_itemsets: pd.DataFrame = mine_frequent_itemsets_mlext_fpgrowth(
        transactions=transactions, min_support=min_support, max_len=max_length
    )
    df_class_based_assoc_rules: pd.DataFrame = class_based_association_rules_for_all_targets(
        df_frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    single_target_cars = convert_mlext_association_rules_dataframe_to_single_target_CARs(
        df_association_rules=df_class_based_assoc_rules)
    return single_target_cars


def mine_MCARs_mlext(
        df_train: pd.DataFrame,
        min_support: float = 0.1, min_confidence: float = 0.5,
        max_length=None
) -> Tuple[List[MCAR], Dict[str, float]]:

    transactions: List[List[str]] = dataframe_to_list_of_transactions(df_train)

    start_fim_time_s = time.time()
    df_frequent_itemsets: pd.DataFrame = mine_frequent_itemsets_mlext_fpgrowth(
        transactions=transactions, min_support=min_support, max_len=max_length
    )
    end_fim_time_s = time.time()

    start_assoc_time = end_fim_time_s
    df_assoc_rules: pd.DataFrame = generate_association_rules_from_frequent_itemsets(
        df_frequent_itemsets=df_frequent_itemsets, min_confidence=min_confidence
    )
    mcars: List[MCAR] = convert_mlext_association_rules_dataframe_to_MCARs(df_assoc_rules)
    end_assoc_time = time.time()
    total_fim_time_s = end_fim_time_s - start_fim_time_s
    total_assoc_time_s = end_assoc_time - start_assoc_time

    timing_info = dict(
        total_fim_time_s=total_fim_time_s,
        total_assoc_time_s=total_assoc_time_s
    )

    return mcars, timing_info


def mine_frequent_itemsets_mlext_fpgrowth(transactions: List[List[str]],
                                          min_support: float, max_len: Optional[int] = None
                                          ) -> pd.DataFrame:
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_transactions = pd.DataFrame(te_ary, columns=te.columns_)
    df_frequent_itemsets: pd.DataFrame = fpgrowth(
        df_transactions, min_support=min_support, max_len=max_len, use_colnames=True)
    return df_frequent_itemsets


def generate_class_based_association_rules_from_frequent_itemsets(
        df_frequent_itemsets: pd.DataFrame,
        target_attribute: str,
        min_confidence: float) -> pd.DataFrame:
    df_assoc_rules = class_based_association_rules(df_frequent_itemsets,
                                                   class_attribute_name=target_attribute,
                                                   attribute_value_separator=attribute_value_separator,
                                                   metric="confidence", min_threshold=min_confidence)
    return df_assoc_rules


def generate_association_rules_from_frequent_itemsets(
        df_frequent_itemsets: pd.DataFrame, min_confidence: float) -> pd.DataFrame:
    df_assoc_rules = association_rules(df_frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return df_assoc_rules


def convert_mlext_association_rules_dataframe_to_MCARs(
        df_association_rules: pd.DataFrame
) -> List[MCAR]:
    mcars = []
    for index, row in df_association_rules.iterrows():
        antecedent_tmp = row[antecedents_column]
        consequent_tmp = row[consequents_column]

        support = row[support_column]
        confidence = row[confidence_column]

        antecedent_items = [EQLiteral(*item.split(attribute_value_separator)) for item in antecedent_tmp]
        consequent_items = [EQLiteral(*item.split(attribute_value_separator)) for item in consequent_tmp]

        antecedent = GeneralizedAntecedent(antecedent_items)
        # antecedent = Antecedent(antecedent_items)
        consequent = MCARConsequent(consequent_items)

        rule = MCAR(antecedent, consequent, support, confidence)
        mcars.append(rule)
    return mcars


def convert_mlext_association_rules_dataframe_to_single_target_CARs(
        df_association_rules: pd.DataFrame
) -> List[ClassAssocationRule]:
    st_cars: List[ClassAssocationRule] = []
    for index, row in df_association_rules.iterrows():
        antecedent_tmp = row[antecedents_column]
        consequent_tmp = row[consequents_column]

        support = row[support_column]
        confidence = row[confidence_column]

        antecedent_items: List[CARItem] = [CARItem(*item.split(attribute_value_separator)) for item in antecedent_tmp]
        consequent_items = [CARItem(*item.split(attribute_value_separator)) for item in consequent_tmp]

        single_consequent_attribute = consequent_items[0].attribute
        single_consequent_value = consequent_items[0].value

        antecedent: CARAntecedent = CARAntecedent(items=antecedent_items)
        consequent: CARConsequent = CARConsequent(attribute=single_consequent_attribute, value=single_consequent_value)

        car: ClassAssocationRule = ClassAssocationRule(antecedent=antecedent,
                                                       consequent=consequent,
                                                       support=support,
                                                       confidence=confidence)
        st_cars.append(car)
    return st_cars
