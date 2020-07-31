from typing import Optional, List, Dict, Set, Iterable

import pandas as pd
from mdrsl.data_structures.rules.multi_target_class_association_rule import MCAR

from mdrsl.data_handling.type_checking_dataframe import type_check_dataframe
from mdrsl.rule_models.multi_target_rule_set_clf_utils.rule_combining_strategy import RuleCombiningStrategy
from mdrsl.rule_models.mids.cover.cover_checher_cached import CachedCoverChecker
from mdrsl.rule_models.mids.cover.cover_checker import CoverChecker
from mdrsl.rule_models.mids.model_fitting.mids_fitting_meta_data import MIDSOptimizationMetaData
from mdrsl.rule_models.mids.objective_function.mids_objective_function_abstract import AbstractMIDSObjectiveFunction
from mdrsl.rule_models.mids.cover.overlap_cacher import CachedOverlapChecker, OverlapChecker
from mdrsl.rule_models.multi_target_rule_set_clf_utils.default_classes import DefaultClassStrategy
from mdrsl.rule_models.mids.mids_classifier import MIDSClassifier
from mdrsl.rule_models.mids.objective_function.mids_objective_function_parameters import ObjectiveFunctionParameters
from mdrsl.rule_models.mids.mids_rule import MIDSRule
from mdrsl.rule_models.mids.mids_ruleset import MIDSRuleSet

from sklearn.metrics import accuracy_score

TargetAttr = str


class MIDSAbstractBase:
    """
    Encapsulates the MIDS algorithm.
    """

    def __init__(self):
        self.classifier: Optional[MIDSClassifier] = None
        self.mids_ruleset: Optional[MIDSRuleSet] = None

        self.cover_checker: Optional[CoverChecker] = None
        self.overlap_checker: Optional[OverlapChecker] = None

        self.objective_function: Optional[AbstractMIDSObjectiveFunction] = None
        self.nb_of_objective_function_calls_necessary_for_training: Optional[int] = None

        # Do we use all dataset attributes as targets, of only a specific set?
        # E.g. only those occuring in the current rule set
        self.use_targets_from_rule_set: bool = False
        self.targets_to_predict: Optional[Set[TargetAttr]] = None

    def fit(self, quant_dataframe,
            use_targets_from_rule_set: bool,
            targets_to_use: Optional[List[str]] = None,
            class_association_rules: Optional[Iterable[MCAR]] = None,
            lambda_array: Optional[List[float]] = None,
            algorithm="RDGS",
            cache_cover_checks=True,
            cache_overlap_checks=True,
            default_class_type: DefaultClassStrategy = DefaultClassStrategy.MAJORITY_VALUE_OVER_WHOLE_TRAINING_SET,
            rule_combination_strategy=RuleCombiningStrategy.WEIGHTED_VOTE,
            debug=True,
            objective_scale_factor=1,
            ):
        """
        Run the MIDS object on a dataset.

        """
        type_check_dataframe(quant_dataframe)

        # --- Make sure the ground rule set is initialized ------------------------------------------------------------
        if self.mids_ruleset is None and class_association_rules is not None:
            ids_rules = list(map(MIDSRule, class_association_rules))  # type: List[MIDSRule]
            mids_ruleset = MIDSRuleSet(ids_rules)
        elif self.mids_ruleset is not None:
            print("using provided mids ruleset and not class association rules")
            mids_ruleset = self.mids_ruleset
        else:
            raise Exception("Neither MCARs or MIDSRules are provided for fitting")

        # --- Use all target or only those in the ruleset? -----------------------------------------------------------
        if targets_to_use is not None:
            targets_to_predict = set(targets_to_use)
            self.targets_to_predict = targets_to_use
        else:
            self.use_targets_from_rule_set = use_targets_from_rule_set
            if use_targets_from_rule_set:
                targets_to_predict: Set[TargetAttr] = mids_ruleset.get_target_attributes()
            else:
                targets_to_predict = set(quant_dataframe.columns)
            self.targets_to_predict = targets_to_predict

        # --- Initialize objective function --------------------------------------------------------------------------
        objective_function_parameters = ObjectiveFunctionParameters(
            all_rules=mids_ruleset, quant_dataframe=quant_dataframe, lambda_array=lambda_array,
            target_attributes=list(targets_to_predict)
        )

        # --- Initialize cover checker and overlap checker ------------------------------------------------------------
        if self.cover_checker is None:
            if cache_cover_checks:
                self.cover_checker = CachedCoverChecker(mids_ruleset, quant_dataframe)
            else:
                self.cover_checker = CoverChecker()
        else:
            print("Reusing previously instantiated cover checker of type", str(type(self.cover_checker)))

        if self.overlap_checker is None:
            if cache_overlap_checks:
                self.overlap_checker = CachedOverlapChecker(mids_ruleset, quant_dataframe, self.cover_checker,
                                                            debug=debug)
            else:
                self.overlap_checker = OverlapChecker(self.cover_checker, debug=debug)
        else:
            print("Reusing previously instantiated overlap checker of type", str(type(self.overlap_checker)))

        # --- Submodular maximization --------------------------------------------------------------------------------

        # if len(mids_ruleset) > 0:
        #     pass
        # else:
        #     warnings.warn("Empty rule list was given")

        solution_set: Set[MIDSRule] = self._optimize(
            objective_function_parameters=objective_function_parameters, algorithm=algorithm,
            objective_scale_factor=objective_scale_factor, debug=debug
        )

        optimization_meta_data: MIDSOptimizationMetaData = MIDSOptimizationMetaData(
            mids_objective_function=self.objective_function,
            optimization_algorithm=algorithm,
            solution_set_size=len(solution_set)
        )

        self.classifier = MIDSClassifier(solution_set, quant_dataframe,
                                         list(targets_to_predict),
                                         optimization_meta_data,
                                         default_class_type, rule_combination_strategy)
        return self

    # ------------------------------------------------------------------------------------------------------------------

    def _optimize(self, objective_function_parameters: ObjectiveFunctionParameters, algorithm: str,
                  objective_scale_factor: float, debug: bool) -> Set[MIDSRule]:
        raise NotImplementedError('abstract method')

    def predict(self, quant_dataframe, target_attribute: str):
        return self.classifier.predict(quant_dataframe, target_attribute)

    def score(self, quant_dataframe: pd.DataFrame,
              target_attributes: List[TargetAttr],
              metric=accuracy_score) -> Dict[TargetAttr, float]:
        target_to_acc: Dict[TargetAttr, float] = {}
        # for target_attribute in quant_dataframe.columns:
        for target_attribute in target_attributes:
            predicted_values = self.predict(quant_dataframe, target_attribute)
            actual_values = quant_dataframe[target_attribute].values
            score = metric(actual_values, predicted_values)
            target_to_acc[target_attribute] = score
        return target_to_acc

    # def score(self, quant_dataframe, metric=accuracy_score):
    #     pred = self.predict(quant_dataframe)
    #     actual = quant_dataframe.dataframe.iloc[:, -1].values
    #
    #     return metric( actual, pred)
    #
    # def _calculate_auc_for_ruleconf(self, quant_dataframe):
    #     conf_pred = self.clf.get_prediction_rules(quant_dataframe)
    #
    #     confidences = []
    #     predicted_classes = []
    #
    #     for conf, predicted_class in conf_pred:
    #         confidences.append(conf)
    #         predicted_classes.append(predicted_class)
    #
    #     actual_classes = quant_dataframe.dataframe.iloc[:, -1].values
    #
    #     actual, pred = encode_label(actual_classes, predicted_classes)
    #
    #     corrected_confidences = []
    #
    #     for conf, predicted_class_label in zip(confidences, pred):
    #         if predicted_class_label == None:
    #             corrected_confidences.append(1)
    #
    #         if predicted_class_label == 0:
    #             corrected_confidences.append(1 - conf)
    #         elif predicted_class_label == 1:
    #             corrected_confidences.append(conf)
    #         else:
    #             raise Exception("Use One-vs-all MIDS classifier for calculating AUC for multiclass problems")
    #
    #     return roc_auc_score(actual, corrected_confidences)
    #
    # def _calcutate_auc_classical(self, quant_dataframe):
    #     pred = self.predict(quant_dataframe)
    #     actual = quant_dataframe.dataframe.iloc[:, -1].values
    #
    #     actual, pred = encode_label(actual, pred)
    #
    #     return roc_auc_score(actual, pred)
    #
    # def score_auc(self, quant_dataframe, confidence_based=False):
    #     if confidence_based:
    #         return self._calculate_auc_for_ruleconf(quant_dataframe)
    #     else:
    #         return self._calcutate_auc_classical(quant_dataframe)
