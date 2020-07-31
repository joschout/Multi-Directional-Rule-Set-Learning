import random
import warnings
from collections import Counter
from enum import Enum, auto
from typing import Set, Dict, List

import numpy as np
import pandas as pd

from mdrsl.rule_models.multi_target_rule_set_clf_utils.rule_combining_strategy import RuleCombinator
from mdrsl.rule_models.mids.mids_rule import MIDSRule

TargetAttr = str


def get_majority_classes_over_whole_training_data(
        target_attributes: List[TargetAttr],
        dataframe_training_data: pd.DataFrame
) -> Dict[TargetAttr, object]:

    df_modes = dataframe_training_data.mode()
    if len(df_modes) != 2:
        warnings.warn("multiple modes found for some attributes in the training data; \n" + str(df_modes))

    default_predictions = {}

    # for column in dataframe_training_data.columns:
    for target_attr in target_attributes:
        default_predictions[target_attr] = df_modes[target_attr].iloc[0]
    return default_predictions


class DefaultClassStrategy(Enum):
    MAJORITY_VALUE_OVER_WHOLE_TRAINING_SET = auto()
    MAJORITY_VALUE_OVER_UNCLASSIFIED_EXAMPLES = auto()

    MINORITY_VALUE_OVER_WHOLE_TRAINING_SET = auto()
    MINORITY_VALUE_OVER_UNCLASSIFIED_EXAMPLES = auto()

    def get_default_classes(self, ruleset: Set[MIDSRule],
                            target_attributes: List[TargetAttr],
                            dataframe_training_data: pd.DataFrame,
                            rule_combinator: RuleCombinator):
        if self == DefaultClassStrategy.MAJORITY_VALUE_OVER_WHOLE_TRAINING_SET:
            return self.__majority_class_over_whole_training_data(
                target_attributes, dataframe_training_data)
        elif self == DefaultClassStrategy.MAJORITY_VALUE_OVER_UNCLASSIFIED_EXAMPLES:
            return self.___minmax_class_in_uncovered_examples(
                ruleset, target_attributes, dataframe_training_data, rule_combinator, to_get='MAJORITY')
        elif self == DefaultClassStrategy.MINORITY_VALUE_OVER_WHOLE_TRAINING_SET:
            return self.__minority_class_over_whole_training_data(target_attributes, dataframe_training_data)
        elif self == DefaultClassStrategy.MINORITY_VALUE_OVER_UNCLASSIFIED_EXAMPLES:
            return self.___minmax_class_in_uncovered_examples(
                ruleset, target_attributes, dataframe_training_data, rule_combinator, to_get='MINORITY')
        else:
            raise Exception("SHOULD NEVER REACH THIS")

    def ___minmax_class_in_uncovered_examples(self,
                                              ruleset: Set[MIDSRule],
                                              target_attributes: List[TargetAttr],
                                              dataframe_training_data: pd.DataFrame,
                                              rule_combinator: RuleCombinator, to_get='MINORITY'):
        default_predictions: Dict[TargetAttr] = {}

        for target_attr in target_attributes:
            default_predictions[target_attr] = self.___minmax_class_in_uncovered_examples_for_target_attibute(
                ruleset, dataframe_training_data, target_attr, rule_combinator, to_get=to_get
            )
        return default_predictions

    def ___minmax_class_in_uncovered_examples_for_target_attibute(self,
                                                                  ruleset: Set[MIDSRule],
                                                                  dataframe_training_data: pd.DataFrame,
                                                                  target_attribute: str,
                                                                  rule_combinator: RuleCombinator,
                                                                  to_get='MINORITY'
                                                                  ):
        """
        Default class = majority class in the uncovered examples


        Examples not covered by ANY rule?
        Examples not covered by a rule predicting a target? I.e. per target attribute, a different set of attributes.

        :return:
        """
        # note: following contains None if no predicted value found
        predicted_class_per_row = rule_combinator.predict(ruleset, dataframe_training_data, target_attribute,
                                                          default_value=None)
        indexes_unclassified_instances = [idx for idx, val in enumerate(predicted_class_per_row) if val == None]
        actual_classes_unclassified_instances = dataframe_training_data[target_attribute].iloc[
            indexes_unclassified_instances]

        # return random class if all examples have a predicted value
        if len(list(actual_classes_unclassified_instances)) == 0:
            distinct_class_values = set(np.unique(dataframe_training_data[target_attribute]))

            default_class = random.sample(
                distinct_class_values, 1
            )[0]
        else:
            counts = Counter(list(actual_classes_unclassified_instances))
            if to_get == 'MAJORITY':
                majority_class = max(counts, key=counts.get)
                default_class = majority_class
            elif to_get == 'MINORITY':
                minority_class = min(counts, key=counts.get)
                default_class = minority_class
            else:
                raise Exception("Should not reach this. Got " + str(to_get) + " as value of to_get parameter.")
        return default_class

    def __majority_class_over_whole_training_data(self,
                                                  # ruleset: Set[MIDSRule],
                                                  target_attributes: List[TargetAttr],
                                                  dataframe_training_data: pd.DataFrame):
        return get_majority_classes_over_whole_training_data(
            target_attributes=target_attributes, dataframe_training_data=dataframe_training_data
        )

    def __minority_class_over_whole_training_data(self,
                                                  # ruleset: Set[MIDSRule],
                                                  target_attributes: List[TargetAttr],
                                                  dataframe_training_data: pd.DataFrame):

        default_predictions = {}

        # for column in dataframe_training_data.columns:
        for target_attr in target_attributes:
            counts = dataframe_training_data[target_attr].value_counts(ascending=True)
            default_predictions[target_attr] = counts.index.values[0]  # datadf_modes[column].iloc[0]
        return default_predictions


if __name__ == '__main__':
    l = ['a', 'b', 'b', 'c', 'c']
    c = Counter(l)
    m = max(c, key=c.get)
    print(c.most_common(1))
    print(m)
