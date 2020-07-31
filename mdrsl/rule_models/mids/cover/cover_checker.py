from typing import List, Optional

import numpy as np
import pandas as pd

from mdrsl.data_structures.item import Literal
from mdrsl.data_structures.rules.rule_part import Antecedent, Consequent, RulePart

from mdrsl.rule_models.mids.mids_rule import MIDSRule

TargetAttr = str


class CoverChecker:

    @staticmethod
    def mask_to_list_of_indexes(mask: np.ndarray, df: pd.DataFrame) -> List[int]:
        return list(df[mask].index.values)  # type: List[int]

    def get_cover(self, rule: MIDSRule, df: pd.DataFrame) -> np.ndarray:
        """
        Finds the covered examples in the data frame,
        i.e. the examples for which the antecedent is true.

        :return: the covered examples as a np.ndarray[bool]
        """

        antecedent = rule.get_antecedent()  # type: Antecedent
        return self._find_cover_mask_rule_part(antecedent, df)

    def get_cover_size(self, rule: MIDSRule, df: pd.DataFrame) -> int:
        return np.sum(self.get_cover(rule, df))

    def _find_cover_mask_rule_part(self, rule_part: RulePart, df: pd.DataFrame) -> np.ndarray:
        dataset_size: int = df.index.size

        # list of ones, all examples covered
        acumulated_mask: np.ndarray = np.ones(dataset_size, dtype=bool)

        literal: Literal
        for literal in rule_part.get_literals():
            # NOTE: value is always a string, as there MUST be discretization of numerical attributes
            # TODO: pyARC also supports Intervals!
            df_attribute_values_np: np.ndarray = df[literal.get_attribute()].to_numpy()
            mask_satisfying_examples: np.ndarray = literal.is_satisfied_for(value_array=df_attribute_values_np)
            acumulated_mask = np.logical_and(acumulated_mask, mask_satisfying_examples)

        # for attribute, value in antecedent.itemset.items():
        #     # NOTE: value is always a string, as there MUST be discretization of numerical attributes
        #     # TODO: pyARC also supports Intervals!
        #     df_attribute_np = df[attribute].to_numpy()  # type: np.ndarray
        #     # comp = df_attribute_np == value
        #     # raise Exception(str(type(comp)) + " " + str(comp))
        #
        #     acumulated_mask = np.logical_and(acumulated_mask, (df_attribute_np == value))
        #     # acumulated_mask &= df[attribute] == value  # type: # np.ndarray

        return acumulated_mask

    def get_correct_cover(self,
                          rule: MIDSRule, df: pd.DataFrame,
                          target_attribute: TargetAttr,
                          cover: Optional[np.ndarray] = None) -> np.ndarray:

        # mask of all points satisfying the rule
        if cover is None:
            cover_mask: np.ndarray = self.get_cover(rule, df)
        else:
            cover_mask = cover

        # ------------

        consequent: Consequent = rule.get_consequent()

        # if target_attribute not in consequent:
        #     raise Exception("the given attribute'", target_attribute, "' is not part of the rule consequent of rule",
        #                     str(rule))
        # predicted_target_value = consequent.get_predicted_value(target_attribute)
        target_literal: Literal = consequent.get_literal(target_attribute)

        # make a series of all target labels
        try:
            target_values_np: np.ndarray = df[target_attribute].to_numpy()
        except Exception as err:
            print("==== KEY ERROR ===")
            print(df.columns)
            print(target_attribute)
            raise err
        mask_correct_target_value: np.ndarray = target_literal.is_satisfied_for(value_array=target_values_np)
        # mask_correct_target_value = target_values_np == predicted_target_value
        correct_cover_mask: np.ndarray = np.logical_and(cover_mask, mask_correct_target_value)
        # print("target_values", target_values_np)
        # print("correct_cover_mask", correct_cover_mask)

        return correct_cover_mask

    def get_incorrect_cover(self,
                            rule: MIDSRule, df: pd.DataFrame,
                            target_attribute: TargetAttr,
                            cover: Optional[np.ndarray] = None
                            ) -> np.ndarray:
        if cover is None:
            full_cover_mask: np.ndarray = self.get_cover(rule, df)
        else:
            full_cover_mask = cover

        consequent: Consequent = rule.get_consequent()
        if target_attribute not in consequent:
            raise Exception("the given attribute'", target_attribute, "' is not part of the rule consequent of rule",
                            str(rule))
        target_literal: Literal = consequent.get_literal(target_attribute)
        # predicted_target_value = consequent.itemset.get(target_attribute)

        # make a series of all target labels
        target_values: np.ndarray = df[target_attribute].to_numpy()
        mask_incorrect_target_value: np.ndarray = np.logical_not(
            target_literal.is_satisfied_for(value_array=target_values))
        # mask_incorrect_target_value = target_values != predicted_target_value
        incorrect_cover_mask = np.logical_and(full_cover_mask, mask_incorrect_target_value)

        # correct_cover_mask = self.get_correct_cover(
        #     rule, df, target_attribute, cover=full_cover_mask)  # type: np.ndarray
        #
        # inverse_correct_cover_mask = np.logical_not(correct_cover_mask)
        #
        # incorrect_cover_mask = np.logical_and(full_cover_mask, inverse_correct_cover_mask)
        return incorrect_cover_mask

    # def overlap(self, rule1: MIDSRule, rule2: MIDSRule, df: pd.DataFrame, target_attribute: str):
    #     """
    #     compute the number of points which are covered both by r1 and r2 w.r.t. data frame df
    #     :param rule1:
    #     :param rule2:
    #     :param df:
    #     :param target_attribute:
    #     :return:
    #     """
    #     if not rule1.head_contains_target_attribute(target_attribute) or not rule2.head_contains_target_attribute(
    #             target_attribute):
    #         raise Exception("WARNING: overlap check where one of the two rules does not contain target attribute",
    #                         str(target_attribute))
    #
    #     if rule1 == rule2:
    #         raise Exception("Dont check a rule with itself for overlap")
    #     return np.logical_and(
    #         self.get_cover(rule1, df),
    #         self.get_cover(rule2, df)
    #     )


if __name__ == '__main__':
    dataframe = pd.DataFrame({'A': [1, 2, 3], 'B': [100, 200, 300]})
    print(dataframe)
    print("---")

    col_a = dataframe['A']
    print('col_a', type(col_a), col_a)
    print("---")

    mask_test = np.array([0, 1, 1], dtype=bool)
    masked_col_a = col_a[mask_test]
    print('masked_col_a', type(masked_col_a), masked_col_a)
    print("----")

    selected_masked_col_a = masked_col_a == 2
    print('selected_masked_col_a', type(selected_masked_col_a), selected_masked_col_a)

    # print(df)
    # arr = np.array([0,1,1], dtype=bool)
    # print(arr)
    # s = df['A'][arr]
    # mask = (s == 2)
    #
    # print('mask', mask)

    # col_a = df['A']
