from typing import Set, List, Optional, Dict

from scipy.stats import hmean

import numpy as np
import pandas as pd
import random

from pyids.data_structures.ids_rule import IDSRule
from pyids.model_selection import mode
from pyarc.qcba.data_structures import QuantitativeDataFrame


TargetValArray = np.ndarray
F1Score = float


class IDSClassifier:

    def __init__(self, rules: List[IDSRule],
                 default_class_strategy: Optional[str] = None, default_class=None):
        self.rules: List[IDSRule] = rules  # sorted by f1 score
        self.default_class_strategy: Optional[str] = default_class_strategy
        self.default_class = default_class

    @staticmethod
    def build_ids_classifier(rules: Set[IDSRule], default_class_strategy: str,
                             quant_dataframe_train) -> 'IDSClassifier':
        sorted_rules = sorted(rules, reverse=True)
        ids_classifier = IDSClassifier(sorted_rules, default_class_strategy=default_class_strategy)

        if default_class_strategy == "majority_class_in_all":
            ids_classifier.default_class = mode(quant_dataframe_train.dataframe.iloc[:, -1])
        elif default_class_strategy == "majority_class_in_uncovered":
            ids_classifier.__calculate_default_class(quant_dataframe_train)

        return ids_classifier

    def __str__(self):
        ostr = (
            f"IDS classifier ({str(len(self.rules))} rules)\n"
            f"\tDefault value strategy: {str(self.default_class_strategy)}\n"
            f"\tDefault value: {str(self.default_class)}\n"
        )

        return ostr

    def __calculate_default_class(self, quant_dataframe_train):
        predicted_classes = self.predict(quant_dataframe_train)
        not_classified_idxes = [idx for idx, val in enumerate(predicted_classes) if val == None]
        classes = quant_dataframe_train.dataframe.iloc[:, -1]

        actual_classes = quant_dataframe_train.dataframe.iloc[not_classified_idxes, -1]

        # return random class
        if not list(actual_classes):
            self.default_class = random.sample(np.unique(classes), 1)[0]

        else:
            minority_class = mode(actual_classes)

            self.default_class = minority_class

    def get_prediction_rules(self, quant_dataframe):
        """
        returns a list of tuples with the ?confidence and ?predicted label
        """
        if type(quant_dataframe) != QuantitativeDataFrame:
            print("Type of quant_dataframe must be QuantitativeDataFrame")

        Y: pd.Series = quant_dataframe.dataframe.iloc[:, -1]

        # f1 score to rule prediction map
        y_pred_dict: Dict[F1Score, TargetValArray] = dict()

        # f1 score to rule map
        rules_f1: Dict[F1Score, IDSRule] = dict()

        rule: IDSRule
        for rule in self.rules:
            conf: float = rule.car.confidence
            sup: float = rule.car.support

            y_pred_per_rule: TargetValArray = rule.predict(quant_dataframe)
            rule_f1_score: F1Score = hmean([conf, sup])

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
                all_NA = np.all(y_pred_array[:, i] == IDSRule.DUMMY_LABEL)
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
                non_na_mask = y_pred_array_datacase != IDSRule.DUMMY_LABEL

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

    def predict(self, quant_dataframe):
        if type(quant_dataframe) != QuantitativeDataFrame:
            print("Type of quant_dataframe must be QuantitativeDataFrame")

        """
        Y = quant_dataframe.dataframe.iloc[:,-1]
        y_pred_dict = dict()

        for rule in self.rules:

            conf = rule.car.confidence
            sup = rule.car.support

            y_pred_per_rule = rule.predict(quant_dataframe)
            rule_f1_score = scipy.stats.hmean([conf, sup])

            y_pred_dict.update({rule_f1_score: y_pred_per_rule})

        y_pred_dict = dict(sorted(y_pred_dict.items(), key=lambda item: item[0], reverse=True))

        # rules in rows, instances in columns
        y_pred_array = np.array(list(y_pred_dict.values()))

        y_pred = []

        minority_classes = []

        if y_pred_dict:
            for i in range(len(Y)):
                all_NA = np.all(y_pred_array[:,i] == IDSRule.DUMMY_LABEL)
                if all_NA:
                    minority_classes.append(Y[i])

            # if the ruleset covers all instances                     
            default_class = Y[0]

            if minority_classes:
                default_class = mode(minority_classes)

            for i in range(len(Y)):
                y_pred_array_datacase = y_pred_array[:,i]
                non_na_mask = y_pred_array_datacase != IDSRule.DUMMY_LABEL

                y_pred_array_datacase_non_na = y_pred_array_datacase[non_na_mask]

                if len(y_pred_array_datacase_non_na) > 0:
                    y_pred.append(y_pred_array_datacase_non_na[0])
                else:
                    y_pred.append(default_class)

            return y_pred

        else:
            y_pred = len(Y) * [mode(Y)]

            return y_pred
        """

        predicted_classes = []

        for _, row in quant_dataframe.dataframe.iterrows():
            appended = False
            for rule in self.rules:
                antecedent_dict = dict(rule.car.antecedent)
                counter = True

                for name, value in row.iteritems():
                    if name in antecedent_dict:
                        rule_value = antecedent_dict[name]

                        counter &= rule_value == value

                if counter:
                    _, predicted_class = rule.car.consequent
                    predicted_classes.append(predicted_class)
                    appended = True
                    break

            if not appended:
                predicted_classes.append(self.default_class)

        return predicted_classes
