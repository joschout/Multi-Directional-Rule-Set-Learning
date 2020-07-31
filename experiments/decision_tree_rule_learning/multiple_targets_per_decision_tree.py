"""
Example of how decision trees predicting multiple targets can be turned into a rule set.
"""
import os
import random
import sys
from itertools import combinations
from typing import List, Dict, Tuple, Set

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from experiments.decision_tree_rule_learning.attribute_grouping import Attr, AttrGroupPartitioning

from mdrsl.data_handling.one_hot_encoding.encoding_book_keeping import EncodingBookKeeper
from mdrsl.rule_generation.decision_tree_conversion.tree_to_rule_set_conversion import (
    convert_decision_tree_to_mids_rule_list)
from mdrsl.data_structures.rules.pretty_printing import mids_mcar_to_pretty_string

from mdrsl.rule_models.mids.mids_rule import MIDSRule
from project_info import project_dir


class DTInfo:
    def __init__(self,
                 original_target_attributes: Tuple[Attr],
                 tree_classifier: DecisionTreeClassifier,
                 ohe_descriptive_attrs: List[Attr],
                 ohe_target_attrs: List[Attr]
                 ):
        self.original_target_attrs: Tuple[Attr] = original_target_attributes
        self.ohe_target_attrs: List[Attr] = ohe_target_attrs

        self.ohe_descriptive_attrs: List[Attr] = ohe_descriptive_attrs

        self.tree_classifier: DecisionTreeClassifier = tree_classifier


def get_ohe_descriptive_and_target_attributes(
        original_target_attr_combo: Tuple[Attr], encoding_book_keeper: EncodingBookKeeper) -> Tuple[List[Attr], List[Attr]]:

    original_target_attr_combo_set: Set[Attr] = set(original_target_attr_combo)

    ohe_target_attrs: List[Attr] = []
    ohe_descriptive_attrs: List[Attr] = []
    for ohe_column in encoding_book_keeper.get_one_hot_encoded_columns():
        if encoding_book_keeper.get_original(ohe_column) in original_target_attr_combo_set:
            ohe_target_attrs.append(ohe_column)
        else:
            ohe_descriptive_attrs.append(ohe_column)

    return ohe_descriptive_attrs, ohe_target_attrs


def get_original_target_attribute_partitioning(
        encoding_book_keeper: EncodingBookKeeper, nb_to_choose: int = 2, random_seed=None
) -> AttrGroupPartitioning:
    original_target_attributes: List[Attr] = list(encoding_book_keeper.get_original_columns())

    if random_seed is not None:
        random.seed(random_seed)
    random.shuffle(original_target_attributes)

    original_target_attribute_groups: List[List[Attr]] = [original_target_attributes[i: i+nb_to_choose]
                                                          for i in range(0, len(original_target_attributes),
                                                                         nb_to_choose)]
    return original_target_attribute_groups


def get_combinations_of_original_target_attributes(
        encoding_book_keeper: EncodingBookKeeper, nb_to_choose: int = 2) -> List[Tuple[Attr]]:
    original_target_attributes: List[Attr] = list(encoding_book_keeper.get_original_columns())

    if nb_to_choose > len(original_target_attributes):
        raise Exception(f"Cannot choose {nb_to_choose} attributes from {len(original_target_attributes)} attributes")
    combos = list(combinations(original_target_attributes, nb_to_choose))
    return combos


def plot_tree(dt_info: DTInfo):
    tree_clf = dt_info.tree_classifier

    class_labels = tree_clf.classes_

    class_labels: List[str] = list(map(str, class_labels))

    fig = plt.gcf()
    fig.set_size_inches(16, 16)
    tree.plot_tree(tree_clf, feature_names=dt_info.ohe_descriptive_attrs, class_names=class_labels)
    plt.show()
    plt.clf()


def main():
    """
    Example of how decision trees predicting multiple targets can be turned into a rule set.
    """
    sys.path.append(os.path.join(project_dir, 'src'))
    sys.path.append(os.path.join(project_dir, 'external/pyARC'))
    sys.path.append(os.path.join(project_dir, 'external/pyIDS'))

    should_plot_trees: bool = True
    nb_of_original_targets_to_predict = 1

    # --- Loading data ------------------------------------------------------------------------------------------------
    dataset_name = 'titanic'

    train_test_dir = os.path.join(project_dir, 'data/interim/' + dataset_name)

    df_train_one_hot_encoded = pd.read_csv(
        os.path.join(train_test_dir, f'{dataset_name}_train_ohe.csv'))
    # df_test_one_hot_encoded = pd.read_csv(
    #     os.path.join(train_test_dir, f'{dataset_name}_test_ohe.csv'))

    # --- creating an encoding ----------------------------------------------------------------------------------------

    encoding_book_keeper = EncodingBookKeeper(ohe_prefix_separator='=')
    encoding_book_keeper.parse_and_store_one_hot_encoded_columns(df_train_one_hot_encoded.columns)

    # --- Learning one decision tree per attribute ---------------------------------------------------------------------

    tree_classifiers: Dict[Tuple[Attr], DTInfo] = {}

    original_target_attr_combo: Tuple[Attr]
    for original_target_attr_combo in get_combinations_of_original_target_attributes(
            encoding_book_keeper, nb_of_original_targets_to_predict):

        # --- Select the descriptive and target attributes -------------------------------------------------------------
        ohe_descriptive_attrs: List[Attr]
        ohe_target_attrs: List[Attr]
        ohe_descriptive_attrs, ohe_target_attrs = get_ohe_descriptive_and_target_attributes(
            original_target_attr_combo, encoding_book_keeper)

        df_train_one_hot_encoded_descriptive = df_train_one_hot_encoded.loc[:, ohe_descriptive_attrs]
        df_train_one_hot_encoded_target = df_train_one_hot_encoded.loc[:, ohe_target_attrs]

        print("original targets:", original_target_attr_combo)
        print("OHE descriptive attrs:")
        print("\t", ohe_descriptive_attrs)
        print("OHE target attrs:")
        print("\t", ohe_target_attrs)

        # --- Fit a decision tree -----------------------------------------------------------------------------------

        tree_clf = DecisionTreeClassifier()
        tree_clf.fit(df_train_one_hot_encoded_descriptive, df_train_one_hot_encoded_target)

        dt_info = DTInfo(original_target_attributes=original_target_attr_combo,
                         tree_classifier=tree_clf,
                         ohe_descriptive_attrs=ohe_descriptive_attrs,
                         ohe_target_attrs=ohe_target_attrs)
        tree_classifiers[original_target_attr_combo] = dt_info

        print("---")

    # --- Convert the fitted decision tree classifiers to rules -------------------------------------------------------
    all_rules: List[MIDSRule] = []

    original_target_attributes: Tuple[Attr]
    dt_info: DTInfo
    for original_target_attributes, dt_info in tree_classifiers.items():
        tree_clf = dt_info.tree_classifier

        if should_plot_trees:
            plot_tree(dt_info)

        list_of_dt_rules = convert_decision_tree_to_mids_rule_list(
            tree_classifier=tree_clf,
            one_hot_encoded_feature_names=dt_info.ohe_descriptive_attrs,
            target_attribute_names=dt_info.ohe_target_attrs,
            encoding_book_keeper=encoding_book_keeper)
        print(f"original target attributes: {original_target_attributes}")
        for rule in list_of_dt_rules:
            print("\t", mids_mcar_to_pretty_string(rule.car))
        print()
        all_rules.extend(list_of_dt_rules)


if __name__ == '__main__':
    main()
