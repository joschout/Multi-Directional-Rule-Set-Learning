
from typing import Optional, Set

from rule_models.ids.ids_classifier import IDSClassifier
from rule_models.ids.ids_overlap_cacher import init_overlap_cacher
from rule_models.ids.objective_function.ids_objective_function_abstract import AbstractIDSObjectiveFunction

from pyarc.qcba.data_structures import QuantitativeDataFrame

from pyids.data_structures import IDSCacher
from pyids.data_structures.ids_objective_function import ObjectiveFunctionParameters
from pyids.data_structures.ids_rule import IDSRule
from pyids.data_structures.ids_ruleset import IDSRuleSet
from pyids.model_selection import calculate_ruleset_statistics, encode_label

from sklearn.metrics import accuracy_score, roc_auc_score


class IDSAbstractBase:
    def __init__(self):
        self.clf: Optional[IDSClassifier] = None
        self.cacher: Optional[IDSCacher] = None
        self.ids_ruleset: Optional[IDSRuleSet] = None
        self.objective_function: Optional[AbstractIDSObjectiveFunction] = None
        self.normalize = True

    def fit(self, quant_dataframe, class_association_rules=None, lambda_array=7 * [1], algorithm="RDGS",
            default_class="majority_class_in_all", debug=True, objective_scale_factor=1):

        type_quant_dataframe = type(quant_dataframe)
        if type_quant_dataframe != QuantitativeDataFrame:
            raise Exception(
                "Type of quant_dataframe must be " + str(QuantitativeDataFrame) + ", but is " +
                str(type_quant_dataframe))

        # init params
        params = ObjectiveFunctionParameters()

        if not self.ids_ruleset:
            ids_rules = list(map(IDSRule, class_association_rules))
            all_rules = IDSRuleSet(ids_rules)
            params.params["all_rules"] = all_rules
        elif self.ids_ruleset and not class_association_rules:
            if debug:
                print("using provided ids ruleset and not class association rules")
            params.params["all_rules"] = self.ids_ruleset

        ground_set = set(params.params["all_rules"].ruleset)

        params.params["len_all_rules"] = len(params.params["all_rules"])
        params.params["quant_dataframe"] = quant_dataframe
        params.params["lambda_array"] = lambda_array

        initialized_cacher = init_overlap_cacher(self.cacher, params.params["all_rules"], quant_dataframe)
        self.cacher = initialized_cacher
        # objective function ------------------------------------------------------------------------------------------
        solution_set: Set[IDSRule] = self._optimize(
            params=params, algorithm=algorithm, ground_set=ground_set,
            cacher=initialized_cacher, objective_scale_factor=objective_scale_factor, debug=debug)
        print("solution set size", len(solution_set))

        # -------------------------------------------------------------------------------------------------------------
        self.clf = IDSClassifier.build_ids_classifier(rules=solution_set,
                                                      default_class_strategy=default_class,
                                                      quant_dataframe_train=quant_dataframe)
        return self

    def _optimize(self, params: ObjectiveFunctionParameters, algorithm: str, ground_set: Set[IDSRule],
                  cacher: IDSCacher, objective_scale_factor: float, debug: bool):
        raise NotImplementedError('abstract method')

    def predict(self, quant_dataframe):
        return self.clf.predict(quant_dataframe)

    def get_prediction_rules(self, quant_dataframe):
        return self.clf.get_prediction_rules(quant_dataframe)

    def score(self, quant_dataframe, metric=accuracy_score):
        pred = self.predict(quant_dataframe)
        actual = quant_dataframe.dataframe.iloc[:, -1].values

        return metric(actual, pred)

    def _calculate_auc_for_ruleconf(self, quant_dataframe):
        conf_pred = self.clf.get_prediction_rules(quant_dataframe)

        confidences = []
        predicted_classes = []

        for conf, predicted_class in conf_pred:
            confidences.append(conf)
            predicted_classes.append(predicted_class)

        actual_classes = quant_dataframe.dataframe.iloc[:, -1].values

        actual, pred = encode_label(actual_classes, predicted_classes)

        corrected_confidences = []

        for conf, predicted_class_label in zip(confidences, pred):
            if predicted_class_label == None:
                corrected_confidences.append(1)

            if predicted_class_label == 0:
                corrected_confidences.append(1 - conf)
            elif predicted_class_label == 1:
                corrected_confidences.append(conf)
            else:
                raise Exception("Use One-vs-all IDS classifier for calculating AUC for multiclass problems")

        return roc_auc_score(actual, corrected_confidences)

    def _calcutate_auc_classical(self, quant_dataframe):
        pred = self.predict(quant_dataframe)
        actual = quant_dataframe.dataframe.iloc[:, -1].values

        actual, pred = encode_label(actual, pred)

        return roc_auc_score(actual, pred)

    def score_auc(self, quant_dataframe, confidence_based=False):
        if confidence_based:
            return self._calculate_auc_for_ruleconf(quant_dataframe)
        else:
            return self._calcutate_auc_classical(quant_dataframe)

    def score_interpretable_metrics(self, quant_dataframe):
        current_ruleset = IDSRuleSet(self.clf.rules)

        stats = calculate_ruleset_statistics(current_ruleset, quant_dataframe)
        return stats
