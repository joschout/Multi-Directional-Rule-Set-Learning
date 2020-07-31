import warnings
from typing import Set

import numpy as np
from pyarc.qcba.data_structures import QuantitativeDataFrame
from pyids.data_structures.ids_rule import IDSRule
from pyids.data_structures.ids_ruleset import IDSRuleSet

from mdrsl.evaluation.interpretability.basic_rule_set_stats import is_valid_fraction, \
    SingleTargetRuleSetStatistics
from utils.value_collection import ValueCollector

TargetVal = object


class IDSInterpretabilityStatistics(SingleTargetRuleSetStatistics):
    def __init__(self,
                 rule_length_collector: ValueCollector,
                 fraction_bodily_overlap: float,
                 fraction_uncovered_examples: float,
                 frac_predicted_classes: float
                 ):
        super().__init__(rule_length_collector, model_abbreviation="IDS",
                         fraction_bodily_overlap=fraction_bodily_overlap,
                         fraction_uncovered_examples=fraction_uncovered_examples,
                         frac_predicted_classes=frac_predicted_classes
                         )

    def to_str(self, indentation: str = "") -> str:
        return super().to_str(indentation=indentation)

    def __str__(self):
        return self.to_str()


class IDSInterpretabilityStatisticsCalculator:

    @staticmethod
    def fraction_bodily_overlap(ruleset: IDSRuleSet, quant_dataframe: QuantitativeDataFrame) -> float:
        """
        This metric captures the extend of overlap between every pair of rules in a decision set R.
        Smaller values of this metric signify higher interpretability.

        Boundary values:
            0.0 if no rules in R overlap:
            1.0 if all data points in are covered by all rules in R.

        NOTE:
            * this is 0.0 for any decision list,
              because their if-else structure ensures that a rule in the list applies only to those data points
                which have not been covered by any of the preceeding rules
            * this is 0.0 for the empty rule set

        :param ruleset:
        :param quant_dataframe:
        :return:
        """
        if type(ruleset) != IDSRuleSet:
            raise Exception("Type of ruleset must be IDSRuleSet")

        if type(quant_dataframe) != QuantitativeDataFrame:
            raise Exception("Type of quant_dataframe must be QuantitativeDataFrame")

        ruleset_size = len(ruleset)
        if ruleset_size == 0:
            print("Warning: the IDS rule set is empty.")
            return 0.0
        nb_of_test_examples: int = quant_dataframe.dataframe.index.size
        if nb_of_test_examples == 0:
            raise Exception("There are no test instances to calculate overlap on")

        rule: IDSRule
        for rule in ruleset.ruleset:
            rule.calculate_cover(quant_dataframe)

        overlap_sum: int = 0
        for i, rule_i in enumerate(ruleset.ruleset):
            for j, rule_j in enumerate(ruleset.ruleset):
                if i <= j:
                    continue
                overlap_sum += np.sum(rule_i.rule_overlap(rule_j, quant_dataframe))

        if overlap_sum == 0:
            warnings.warn("overlap is 0, which is unlikely")
            return 0
        else:
            frac_overlap: float = 2 / (ruleset_size * (ruleset_size - 1)) * overlap_sum / nb_of_test_examples

            if not is_valid_fraction(frac_overlap):
                raise Exception("Fraction overlap is not within [0,1]: " + str(frac_overlap))

            return frac_overlap

    @staticmethod
    def fraction_uncovered_examples(ruleset: IDSRuleSet, quant_dataframe: QuantitativeDataFrame) -> float:
        """

        This metric computes the fraction of the data points which are not covered by any rule in the decision set.
        REMEMBER, COVER is independent of the head of a rule.

        Boundary values:
            0.0 if every data point is covered by some rule in the data set.
            1.0 when no data point is covered by any rule in R
                (This could be the case when |R| = 0 )

        Note:
            * for decision lists, this metric is
                the fraction of the data points that are covered by the ELSE clause of the list
                (i.e. the default prediction).

        :param ruleset:
        :param quant_dataframe::
        :return:
        """

        if type(ruleset) != IDSRuleSet:
            raise Exception("Type of ruleset must be IDSRuleSet")

        if type(quant_dataframe) != QuantitativeDataFrame:
            raise Exception("Type of quant_dataframe must be QuantitativeDataFrame")

        nb_of_test_examples = quant_dataframe.dataframe.index.size
        if nb_of_test_examples == 0:
            raise Exception("There are no test instances to calculate the fraction uncovered on")

        cover_cumulative_mask: np.ndarray = np.zeros(nb_of_test_examples, dtype=bool)

        for rule in ruleset.ruleset:
            cover_mask = rule._cover(quant_dataframe)
            cover_cumulative_mask = np.logical_or(cover_cumulative_mask, cover_mask)

        nb_of_covered_test_examples: int = np.count_nonzero(cover_cumulative_mask)

        frac_uncovered = 1 - 1 / nb_of_test_examples * nb_of_covered_test_examples

        if not is_valid_fraction(frac_uncovered):
            raise Exception("Fraction uncovered examples is not within [0,1]: " + str(frac_uncovered))

        return frac_uncovered

    @staticmethod
    def fraction_predicted_classes(ruleset: IDSRuleSet, quant_dataframe: QuantitativeDataFrame) -> float:
        """
        This metric denotes the fraction of the classes in the data that are predicted by the ruleset R.

        Boundary value:
            0.0 if no class is predicted (e.g. the ruleset is empty)
            1.0 every class is predicted by some rule in R.

        Note:
            * The same for decision lists, but we not consider the ELSE clause (the default prediction).

        :param ruleset:
        :param quant_dataframe:
        :return:
        """
        if type(ruleset) != IDSRuleSet:
            raise Exception("Type of ruleset must be IDSRuleSet")

        if type(quant_dataframe) != QuantitativeDataFrame:
            raise Exception("Type of quant_dataframe must be QuantitativeDataFrame")

        values_occuring_in_data: Set[TargetVal] = set(quant_dataframe.dataframe.iloc[:, -1].values)
        values_predicted_by_rules: Set[TargetVal] = set()

        for rule in ruleset.ruleset:
            covered_class: TargetVal = rule.car.consequent.value
            if covered_class in values_occuring_in_data:
                values_predicted_by_rules.add(covered_class)

        nb_of_values_predicted_by_rules = len(values_predicted_by_rules)
        nb_of_values_occuring_in_test_data = len(values_occuring_in_data)
        
        frac_classes: float = nb_of_values_predicted_by_rules / nb_of_values_occuring_in_test_data

        if not is_valid_fraction(frac_classes):
            raise Exception("Fraction predicted classes examples is not within [0,1]: " + str(frac_classes))

        return frac_classes

    @staticmethod
    def calculate_ruleset_statistics(ruleset: IDSRuleSet,
                                     quant_dataframe: QuantitativeDataFrame) -> IDSInterpretabilityStatistics:

        rule_length_collector = ValueCollector()
        for rule in ruleset.ruleset:
            rule_length_collector.add_value(len(rule))

        fraction_bodily_overlap: float = IDSInterpretabilityStatisticsCalculator.fraction_bodily_overlap(
            ruleset=ruleset, quant_dataframe=quant_dataframe)
        fraction_uncovered_examples: float = IDSInterpretabilityStatisticsCalculator.fraction_uncovered_examples(
            ruleset=ruleset, quant_dataframe=quant_dataframe)
        frac_predicted_classes: float = IDSInterpretabilityStatisticsCalculator.fraction_predicted_classes(
            ruleset=ruleset, quant_dataframe=quant_dataframe)

        statistics = IDSInterpretabilityStatistics(
                     rule_length_collector=rule_length_collector,
                     fraction_bodily_overlap=fraction_bodily_overlap,
                     fraction_uncovered_examples=fraction_uncovered_examples,
                     frac_predicted_classes=frac_predicted_classes
                     )
        return statistics
