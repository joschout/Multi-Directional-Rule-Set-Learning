from scipy.stats import hmean
from typing import Set, Dict, List, Optional

import numpy as np
import pandas as pd

from mdrsl.data_structures.quant_dataset import QuantitativeDataFrame
from mdrsl.rule_models.multi_target_rule_set_clf_utils.rule_combining_strategy import RuleCombiningStrategy, RuleCombinator
from mdrsl.rule_models.multi_target_rule_set_clf_utils.default_classes import DefaultClassStrategy
from mdrsl.rule_models.mids.mids_rule import MIDSRule
from mdrsl.rule_models.mids.model_fitting.mids_fitting_meta_data import MIDSOptimizationMetaData

TargetAttr = str


class MIDSClassifier:
    """
    Based on MIDS, there are two choices when using the rules for prediction:
    1. how to combine the rules for prediction.
    2. the default predicted value for each attribute, when no prediction is possible in 1.

    Some possible choices are the following:
    1. how to combine the rules for prediction.
        a. Sort the rules according to their f1-score with respect to the target attribute,
           and use the first rule predicting the target that matches the test instance
        b. use a majority vote for the rules that match the instance
    2. the default predicted value for an attribute:
        a. predict the majority target value over the whole training data
        b. predict the smallest minority class on the whole data
        c. predict the majority target value over the training instances that are not covered by the rules predicting the target.
    """

    def __init__(self, rules: Set[MIDSRule], df_training_data: pd.DataFrame,
                 target_attributes: List[TargetAttr],
                 optimization_meta_data: Optional[MIDSOptimizationMetaData],
                 default_class_type: DefaultClassStrategy = DefaultClassStrategy.MAJORITY_VALUE_OVER_WHOLE_TRAINING_SET,
                 rule_combination_strategy=RuleCombiningStrategy.WEIGHTED_VOTE,
                 ):
        if rule_combination_strategy == RuleCombiningStrategy.F1_SCORE:
            self.rules: List[MIDSRule] = sorted(rules, reverse=True)
        else:
            self.rules: Set[MIDSRule] = rules

        self.target_attrs: List[TargetAttr] = target_attributes
        self.optimization_meta_data: Optional[MIDSOptimizationMetaData] = optimization_meta_data

        self.rule_combinator: RuleCombinator = rule_combination_strategy.create()
        self.rule_combination_strategy: RuleCombiningStrategy = rule_combination_strategy

        self.default_predictions: Dict[TargetAttr, object] = default_class_type.get_default_classes(
            rules, target_attributes, df_training_data, self.rule_combinator)
        self.default_class_strategy: DefaultClassStrategy = default_class_type

    def __str__(self):
        ostr = (
                "MIDS classifier (" + str(len(self.rules)) + " rules)\n"
                + "\tRule combination stategy: " + str(self.rule_combination_strategy) + "\n"
                + "\tDefault value strategy: " + str(self.default_class_strategy) + "\n"
                + "\t\tDefault predictions:" + str(self.default_predictions) + "\n"
        )

        return ostr

    def predict(self, dataframe: pd.DataFrame, target_attribute: str):
        return self.rule_combinator.predict(
            self.rules, dataframe, target_attribute, self.default_predictions[target_attribute])

    def get_prediction_rules(self, quant_dataframe):
        if type(quant_dataframe) != QuantitativeDataFrame:
            print("Type of quant_dataframe must be QuantitativeDataFrame")

        Y = quant_dataframe.dataframe.iloc[:, -1]
        y_pred_dict = dict()
        rules_f1 = dict()

        for rule in self.rules:
            conf = rule.car.confidence
            sup = rule.car.support

            y_pred_per_rule = rule.predict(quant_dataframe)
            rule_f1_score = hmean([conf, sup])

            y_pred_dict.update({rule_f1_score: y_pred_per_rule})
            rules_f1.update({rule_f1_score: rule})

        # rules in rows, instances in columns
        y_pred_array = np.array(list(y_pred_dict.values()))

        y_pred_dict = dict(sorted(y_pred_dict.items(), key=lambda item: item[0], reverse=True))

        y_pred = []

        minority_classes = []

        rule_list = list(self.rules)

        if y_pred_dict:
            for i in range(len(Y)):
                all_NA = np.all(y_pred_array[:, i] == MIDSRule.DUMMY_LABEL)
                if all_NA:
                    minority_classes.append(Y[i])

            # if the ruleset covers all instances
            default_class = len(Y == Y[0]) / len(Y)
            default_class_label = Y[0]

            if minority_classes:
                default_class = len(Y == mode(minority_classes)) / len(Y)
                default_class_label = mode(minority_classes)

            for i in range(len(Y)):
                y_pred_array_datacase = y_pred_array[:, i]
                non_na_mask = y_pred_array_datacase != MIDSRule.DUMMY_LABEL

                y_pred_array_datacase_non_na = np.where(non_na_mask)[0]

                if len(y_pred_array_datacase_non_na) > 0:
                    rule_index = y_pred_array_datacase_non_na[0]
                    rule = rule_list[rule_index]

                    y_pred.append((rule.car.confidence, rule.car.consequent.value))
                else:
                    y_pred.append((default_class, default_class_label))

            return y_pred

        else:
            y_pred = len(Y) * [np.inf]

            return y_pred


def mode(array):
    values, counts = np.unique(array, return_counts=True)
    idx = np.argmax(counts)

    return values[idx]
