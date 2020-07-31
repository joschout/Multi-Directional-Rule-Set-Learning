import warnings
from typing import Set, Dict, Optional, List, Tuple

import numpy as np
import pandas as pd

from mdrsl.data_structures.rules.rule_part import Consequent
from mdrsl.evaluation.interpretability.basic_rule_set_stats import BasicRuleSetStatistics, is_valid_fraction
from mdrsl.rule_models.mids.cover.cover_checker import CoverChecker
from mdrsl.rule_models.mids.cover.overlap_cacher import OverlapChecker
from mdrsl.rule_models.mids.mids_rule import MIDSRule
from mdrsl.rule_models.mids.mids_ruleset import MIDSRuleSet
from mdrsl.utils.value_collection import ValueCollector

TargetAttr = str
TargetVal = object

DefaultCoverChecker = CoverChecker
DefaultOverlapChecker = OverlapChecker


class MIDSInterpretabilityStatistics(BasicRuleSetStatistics):
    def __init__(self,
                 rule_length_collector: ValueCollector,
                 fraction_bodily_overlap: float,
                 fraction_uncovered_examples: float,
                 avg_frac_predicted_classes: float,
                 frac_predicted_classes_per_target_attr: Dict[TargetAttr, float],
                 # ground_set_size: Optional[int] = None
                 ):
        super().__init__(rule_length_collector, model_abbreviation="MIDS")
        self.fraction_bodily_overlap: float = fraction_bodily_overlap
        self.fraction_uncovered_examples: float = fraction_uncovered_examples
        self.avg_frac_predicted_classes: float = avg_frac_predicted_classes
        self.frac_predicted_classes_per_target_attr: Dict[TargetAttr, float] = frac_predicted_classes_per_target_attr

        # self.ground_set_size: Optional[int] = ground_set_size

    def to_str(self, indentation: str = "") -> str:
        output_string = (
                indentation + "Rule length stats: " + str(self.rule_length_counter) + "\n"
                + indentation + "Fraction bodily overlap: " + str(self.fraction_bodily_overlap) + "\n"
                + indentation + "Fraction uncovered examples: " + str(self.fraction_uncovered_examples) + "\n"
                + indentation + "Avg fraction predicted classes: " + str(self.avg_frac_predicted_classes) + "\n"
                + indentation + "Fraction predicted classs by target:\n"
                + indentation + "\t" + str(self.frac_predicted_classes_per_target_attr) + "\n"
        )

        return output_string

    def __str__(self):
        return self.to_str()

    def satisfies(self, condition: 'MIDSInterpretabilityStatitisticsAbstractCondition') -> bool:
        return condition.is_satisfied_by(self)


class MIDSInterpretabilityStatisticsCalculator:

    @staticmethod
    def _fraction_overlap(ruleset: MIDSRuleSet, test_dataframe: pd.DataFrame,
                          target_attr: Optional[TargetAttr] = None,
                          cover_checker_on_test: Optional[CoverChecker] = None,
                          overlap_checker_on_test: Optional[OverlapChecker] = None,
                          debug=False) -> float:
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
        :param test_dataframe:
        :param cover_checker_on_test:
        :param overlap_checker_on_test:
        :return:
        """

        if type(ruleset) != MIDSRuleSet:
            raise Exception(f"Type of ruleset must be MIDSRuleSet, but is {type(ruleset)}")
        warnings.warn("FRACTION_OVERLAP IS CURRENTLY NOT RELATIVE TO A TARGET ATTRIBUTE. THIS MIGHT BE INCORRECT")

        ruleset_size: int = len(ruleset)
        if ruleset_size == 0:
            print("Warning: the MIDS rule set is empty.")
            return 0.0
        nb_of_test_examples: int = test_dataframe.index.size
        if nb_of_test_examples == 0:
            raise Exception("There are no test instances to calculate overlap on")

        if cover_checker_on_test is None:
            cover_checker_on_test = DefaultCoverChecker()
        if overlap_checker_on_test is None:
            overlap_checker_on_test = DefaultOverlapChecker(cover_checker_on_test, debug)

        overlap_sum: int = 0

        rule_i: MIDSRule
        rule_j: MIDSRule
        for i, rule_i in enumerate(ruleset.ruleset):
            for j, rule_j in enumerate(ruleset.ruleset):
                if i <= j:
                    continue
                else:
                    if target_attr is None:
                        overlap_sum += overlap_checker_on_test.get_pure_overlap_count(rule_i, rule_j, test_dataframe)
                    else:
                        overlap_sum += overlap_checker_on_test.get_relative_overlap_count(rule_i, rule_j,
                                                                                          test_dataframe, target_attr)

        if overlap_sum == 0:
            warnings.warn("overlap is 0, which is unlikely")
            return 0
        else:
            frac_overlap = 2 / (ruleset_size * (ruleset_size - 1)) * overlap_sum / nb_of_test_examples

            if not is_valid_fraction(frac_overlap):
                raise Exception("Fraction overlap is not within [0,1]: " + str(frac_overlap))

            return frac_overlap

    @staticmethod
    def fraction_bodily_overlap(ruleset: MIDSRuleSet, test_dataframe: pd.DataFrame,
                                cover_checker_on_test: Optional[CoverChecker] = None,
                                overlap_checker_on_test: Optional[OverlapChecker] = None,
                                debug=False) -> float:
        return MIDSInterpretabilityStatisticsCalculator._fraction_overlap(
            ruleset=ruleset, test_dataframe=test_dataframe,
            target_attr=None,
            cover_checker_on_test=cover_checker_on_test,
            overlap_checker_on_test=overlap_checker_on_test,
            debug=debug
        )

    @staticmethod
    def get_fraction_overlap_relative_to_target_attr(ruleset: MIDSRuleSet, test_dataframe: pd.DataFrame,
                                                     target_attr: str,
                                                     cover_checker_on_test: Optional[CoverChecker] = None,
                                                     overlap_checker_on_test: Optional[OverlapChecker] = None,
                                                     debug=False
                                                     ) -> float:
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
        :param test_dataframe:
        :param target_attr:
        :param cover_checker_on_test:
        :param overlap_checker_on_test:
        :param debug:
        :return:
        """
        if target_attr is None:
            raise Exception("Cannot calculate relative overlap fraction without a given target attr. It is None.")
        return MIDSInterpretabilityStatisticsCalculator._fraction_overlap(
            ruleset=ruleset,
            test_dataframe=test_dataframe,
            target_attr=target_attr,
            cover_checker_on_test=cover_checker_on_test,
            overlap_checker_on_test=overlap_checker_on_test,
            debug=debug
        )

    @staticmethod
    def fraction_uncovered_examples(ruleset: MIDSRuleSet, test_dataframe: pd.DataFrame,
                                    cover_checker_on_test: Optional[CoverChecker] = None
                                    ) -> float:
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
        :param test_dataframe:
        :param cover_checker_on_test:
        :return:
        """
        if type(ruleset) != MIDSRuleSet:
            raise Exception("Type of ruleset must be IDSRuleSet")

        if cover_checker_on_test is None:
            cover_checker_on_test = DefaultCoverChecker()

        nb_of_test_examples: int = test_dataframe.index.size
        if nb_of_test_examples == 0:
            raise Exception("There are no test instances to calculate the fraction uncovered on")

        cover_cumulative_mask: np.ndarray = np.zeros(nb_of_test_examples, dtype=bool)

        for rule in ruleset.ruleset:
            cover_mask = cover_checker_on_test.get_cover(rule, test_dataframe)
            cover_cumulative_mask = np.logical_or(cover_cumulative_mask, cover_mask)

        nb_of_covered_test_examples: int = np.count_nonzero(cover_cumulative_mask)

        frac_uncovered: float = 1 - 1 / nb_of_test_examples * nb_of_covered_test_examples

        if not is_valid_fraction(frac_uncovered):
            raise Exception("Fraction uncovered examples is not within [0,1]: " + str(frac_uncovered))

        return frac_uncovered

    @staticmethod
    def fraction_predicted_classes(ruleset: MIDSRuleSet, test_dataframe,
                                   target_attributes: List[TargetAttr]
                                   ) -> Tuple[float, Dict[TargetAttr, float]]:
        """
        This metric denotes the fraction of the classes in the data that are predicted by the ruleset R.

        Returns:
            1. fraction per target attribute, averaged over the different targets
            2. a map from target attribute to fraction for that target attr


        Boundary value:
            0.0 if no class is predicted (e.g. the ruleset is empty)
            1.0 every class is predicted by some rule in R.

        Note:
            * The same for decision lists, but we not consider the ELSE clause (the default prediction).

        :param target_attributes:
        :param ruleset:
        :param test_dataframe:
        :return:
        """

        if type(ruleset) != MIDSRuleSet:
            raise Exception("Type of ruleset must be IDSRuleSet")

        warnings.warn(
            "Ugly conversion to string to deal with numerical attributes."
            " Clean this up (look at Survived in Titanic).")

        values_in_data_per_target_attribute: Dict[TargetAttr, Set[TargetVal]] = {}
        predicted_values_per_target_attribute: Dict[TargetAttr, Set[TargetVal]] = {}

        for target_attr in target_attributes:
            values_as_str: List[str] = [str(val) for val in test_dataframe[target_attr].values]
            values_in_data_per_target_attribute[target_attr] = set(values_as_str)
            predicted_values_per_target_attribute[target_attr] = set()

        target_attribute_set: Set[TargetAttr] = set(target_attributes)

        for rule in ruleset.ruleset:
            consequent: Consequent = rule.car.consequent
            for literal in consequent.get_literals():
                predicted_attribute: TargetAttr = literal.get_attribute()
                predicted_value: TargetVal = literal.get_value()

                if predicted_attribute in target_attribute_set:
                    predicted_value_str = str(predicted_value)
                    predicted_values: Set[TargetVal] = predicted_values_per_target_attribute[predicted_attribute]
                    if predicted_value_str in values_in_data_per_target_attribute[predicted_attribute]:
                        predicted_values.add(predicted_value_str)

        # print("values_in_data_per_target_attribute", values_in_data_per_target_attribute)
        # print("predicted_values_per_target_attribute", predicted_values_per_target_attribute)

        frac_predicted_classes_per_target_attr: Dict[TargetAttr, float] = {}

        avg_frac_predicted_classes: float = 0
        for target_attr in values_in_data_per_target_attribute.keys():
            values_occuring_in_data = values_in_data_per_target_attribute[target_attr]
            predicted_values = predicted_values_per_target_attribute[target_attr]

            domain_size_in_test_data = len(values_occuring_in_data)
            nb_of_predicted_values = len(predicted_values)

            frac_classes: float = nb_of_predicted_values / domain_size_in_test_data
            frac_predicted_classes_per_target_attr[target_attr] = frac_classes

            avg_frac_predicted_classes += frac_classes

        nb_of_target_attrs = len(target_attributes)
        avg_frac_predicted_classes = avg_frac_predicted_classes / nb_of_target_attrs

        if not is_valid_fraction(avg_frac_predicted_classes):
            raise Exception("Avg fraction predicted classes examples is not within [0,1]: "
                            + str(avg_frac_predicted_classes))

        return avg_frac_predicted_classes, frac_predicted_classes_per_target_attr

    @staticmethod
    def calculate_ruleset_statistics(ruleset: MIDSRuleSet, test_dataframe: pd.DataFrame,
                                     target_attributes: List[TargetAttr]
                                     ) -> MIDSInterpretabilityStatistics:

        rule_length_collector = ValueCollector()
        for rule in ruleset.ruleset:
            rule_length_collector.add_value(len(rule))

        fraction_bodily_overlap: float = MIDSInterpretabilityStatisticsCalculator.fraction_bodily_overlap(
            ruleset=ruleset, test_dataframe=test_dataframe)
        fraction_uncovered_examples: float = MIDSInterpretabilityStatisticsCalculator.fraction_uncovered_examples(
            ruleset=ruleset, test_dataframe=test_dataframe
        )
        avg_frac_predicted_classes: float
        frac_predicted_classes_per_target_attr: Dict[TargetAttr, float]
        avg_frac_predicted_classes, frac_predicted_classes_per_target_attr = \
            MIDSInterpretabilityStatisticsCalculator.fraction_predicted_classes(
                ruleset=ruleset, test_dataframe=test_dataframe, target_attributes=target_attributes
            )

        statistics = MIDSInterpretabilityStatistics(
            rule_length_collector=rule_length_collector,
            fraction_bodily_overlap=fraction_bodily_overlap,
            fraction_uncovered_examples=fraction_uncovered_examples,
            avg_frac_predicted_classes=avg_frac_predicted_classes,
            frac_predicted_classes_per_target_attr=frac_predicted_classes_per_target_attr
            # ground_set_size=ground_set_size
        )

        return statistics


class MIDSInterpretabilityStatisticsAbstractCondition:
    def is_satisfied_by(self, interpret_stats: MIDSInterpretabilityStatistics) -> bool:
        raise NotImplementedError("abstract method")


class MIDSInterpretabilityStatisticsBoundaryCondition(MIDSInterpretabilityStatisticsAbstractCondition):

    def __init__(self,
                 max_fraction_bodily_overlap: float = 1.0,
                 max_fraction_uncovered_examples: float = 1.0,
                 min_avg_fraction_predicted_classes: float = 0.0,
                 min_frac_predicted_classes_for_each_target_attr: float = 0.5,

                 max_n_rules: int = 500,
                 max_avg_rule_length: int = 10
                 ):

        self.max_fraction_bodily_overlap: float = max_fraction_bodily_overlap
        self.max_fraction_uncovered_examples: float = max_fraction_uncovered_examples

        self.min_avg_fraction_predicted_classes: float = min_avg_fraction_predicted_classes
        self.min_frac_predicted_classes_for_each_target_attr: float = min_frac_predicted_classes_for_each_target_attr

        self.max_n_rules: int = max_n_rules
        self.max_avg_rule_length: int = max_avg_rule_length

    def is_satisfied_by(self, interpret_stats: MIDSInterpretabilityStatistics) -> bool:
        return (
                interpret_stats.ruleset_size() <= self.max_n_rules and
                interpret_stats.avg_nb_of_literals_per_rule() <= self.max_avg_rule_length and

                interpret_stats.fraction_bodily_overlap <= self.max_fraction_bodily_overlap and
                interpret_stats.fraction_uncovered_examples <= self.max_fraction_uncovered_examples and

                interpret_stats.avg_frac_predicted_classes >= self.min_avg_fraction_predicted_classes and
                self._is_frac_predicted_classes_for_each_class_above_threshold(interpret_stats)

        )

    def _is_frac_predicted_classes_for_each_class_above_threshold(self,
                                                                  interpret_stats: MIDSInterpretabilityStatistics):
        for frac_predicted_classes_for_single_target in interpret_stats.frac_predicted_classes_per_target_attr.values():
            if frac_predicted_classes_for_single_target < self.min_frac_predicted_classes_for_each_target_attr:
                return False

        return True
