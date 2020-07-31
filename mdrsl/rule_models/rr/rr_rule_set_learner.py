import time
from typing import Set, Optional, List, Dict, Tuple

import pandas as pd
from sklearn.metrics import f1_score

from mdrsl.data_handling.type_checking_dataframe import type_check_dataframe
from mdrsl.data_handling.nan_data_filtering import remove_instances_with_nans_in_column

from mdrsl.rule_models.multi_target_rule_set_clf_utils.rule_combining_strategy import RuleCombinator, RuleCombiningStrategy
from mdrsl.rule_models.multi_target_rule_set_clf_utils.default_classes import get_majority_classes_over_whole_training_data
from mdrsl.rule_models.mids.mids_rule import MIDSRule

ElemScore = float
RuleSetScore = float

TargetAttr = str


class GreedyRoundRobinTargetRuleClassifier:

    def __init__(self, target_attributes: List[TargetAttr], verbose: bool = False,
                 keep_score_evolution=True,
                 min_required_score_increase: float = 0.01,
                 max_score_diff_with_best_rule: float = 0.1
                 ):
        if len(target_attributes) == 0:
            raise Exception("Need at least one target attribute")
        self.target_attributes: List[TargetAttr] = list(target_attributes)
        self.current_target_attr_index: int = len(self.target_attributes)

        self.rule_combinator: Optional[RuleCombinator] = None
        self.rule_combination_strategy: Optional[RuleCombiningStrategy] = None

        self.default_predictions: Optional[Dict[TargetAttr, object]] = None

        self.targets_to_check: Dict[TargetAttr, bool] = {}
        self._set_all_targets_must_be_checked()

        self.verbose: bool = verbose
        self.keep_score_evolution: bool = keep_score_evolution
        self.score_evolution: Optional[Dict[TargetAttr, List[RuleSetScore]]] = None

        # to be put in meta data:
        self.max_score_diff_with_best_rule: float = max_score_diff_with_best_rule  # delta
        self.min_required_score_increase: float = min_required_score_increase  # epsilon

        self.ground_set_size: Optional[int] = None

        self.learned_rule_set: Optional[Set[MIDSRule]] = None
        self.learned_rule_set_scores: Optional[Dict[TargetAttr, RuleSetScore]] = None
        self.learning_time_s: Optional[float] = None

        self.filter_nans: bool = True
        self.should_cache_nan_filtered_training_data: bool = True
        self.nan_filterered_training_data_cache: Optional[TargetAttr, pd.DataFrame] = None

    def __str__(self):
        ostr = (
                "Greedy Naive Round Robin classifier (" + str(len(self.learned_rule_set)) + " rules)\n"
                + "\tRule combination stategy: " + str(self.rule_combinator.__class__) + "\n"
                + "\tDefault predictions:\n"
        )
        self.target_attributes.sort()

        max_str_len_target = 0
        max_str_len_val = 0
        for target, val in self.default_predictions.items():
            target_str_len = len(str(target))
            if target_str_len > max_str_len_target:
                max_str_len_target = target_str_len
            val_str_len = len(str(val))
            if val_str_len > max_str_len_val:
                max_str_len_val = val_str_len

        for target in self.target_attributes:
            ostr += (
                    "\t\t" + f"{target}".ljust(max_str_len_target) + ": "
                    + f"{self.default_predictions[target]}".ljust(max_str_len_val)
                    + f" ({self.learned_rule_set_scores[target]:.2f})\n"
            )

        return ostr

    def __repr__(self):
        return self.__str__()

    def _init_score_evolution(self):
        self.score_evolution = {}
        for target in self.target_attributes:
            self.score_evolution[target] = []

    def _add_to_score_evolution(self, scores_of_rule_set: Dict[TargetAttr, RuleSetScore]) -> None:
        for target in self.target_attributes:
            self.score_evolution[target].append(scores_of_rule_set[target])

    def score_evolution_to_dataframe(self) -> Optional[pd.DataFrame]:
        if self.keep_score_evolution:
            columns = ['time_point', 'target_attribute', 'score_value']
            data = []

            for target, score_list in self.score_evolution.items():
                for time_point, score in enumerate(score_list):
                    data.append([time_point, target, score])
            df = pd.DataFrame(data=data, columns=columns)
            return df

        else:
            return None

    def _set_all_targets_must_be_checked(self):
        for target in self.target_attributes:
            self.targets_to_check[target] = True

    def _is_target_left_to_check(self) -> bool:
        return any(self.targets_to_check.values())

    def _change_target(self) -> TargetAttr:
        self.current_target_attr_index: int = (self.current_target_attr_index + 1) % len(self.target_attributes)
        return self.target_attributes[self.current_target_attr_index]

    def _get_next_target_to_check(self):
        found_target = False

        n_checks_done = 0
        while not found_target:
            target = self._change_target()
            if self.targets_to_check[target]:
                return target
            n_checks_done += 1
            if n_checks_done > len(self.target_attributes):
                raise Exception()

    def _prepare_cached_training_data(self, training_data: pd.DataFrame):
        self.nan_filterered_training_data_cache = {}
        for target_attribute in self.target_attributes:
            self.nan_filterered_training_data_cache[target_attribute] = remove_instances_with_nans_in_column(
                training_data, target_attribute)

    def _score(self, rules: Set[MIDSRule], training_data: pd.DataFrame,
               target_attribute: TargetAttr):

        if self.filter_nans:
            if self.should_cache_nan_filtered_training_data:
                filtered_test_dataframe = self.nan_filterered_training_data_cache[target_attribute]
            else:
                filtered_test_dataframe = remove_instances_with_nans_in_column(training_data, target_attribute)
        else:
            filtered_test_dataframe = training_data

        predicted_values = self.rule_combinator.predict(
            rules, filtered_test_dataframe, target_attribute, default_value=self.default_predictions[target_attribute])

        actual_values = filtered_test_dataframe[target_attribute].values

        micro_avged_f1_score_value: float = f1_score(actual_values, predicted_values, average='micro')
        return micro_avged_f1_score_value

    def get_best_improvement_for_target(self,
                                        training_data: pd.DataFrame,
                                        current_set: Set[MIDSRule],
                                        current_set_scores: Dict[TargetAttr, RuleSetScore],
                                        ground_set: Set[MIDSRule],
                                        target_attribute: TargetAttr
                                        ) -> Optional[Tuple[Set[MIDSRule],
                                                            Dict[TargetAttr, RuleSetScore],
                                                            List[TargetAttr]]]:
        """
        Find the the rule that improves the current set the most for the given target attr,
        without damaging the other attributes.
        
        :param training_data:
        :param current_set_scores:
        :param current_set:
        :param ground_set: 
        :param target_attribute: 
        :return: 
        """
        # for each candidate rule:
        #   check if:
        #       * it predicts the target
        #       * adding the rule increases the score of the total rule set
        #   IF SO:
        #       * find the scores of the extended rule set on the other targets (OF THE RULE ??)
        target_scores_per_rule: List[Tuple[RuleSetScore, MIDSRule]] = []

        tmp_extended_rule_set = current_set.copy()

        candidate_rule: MIDSRule
        for candidate_rule in ground_set:
            if candidate_rule not in current_set:
                rule_target_attrs = candidate_rule.get_target_attributes()
                if target_attribute not in rule_target_attrs:
                    pass
                else:
                    tmp_extended_rule_set.add(candidate_rule)
                    # # NOTE: the nb of examples for which this is incorrect?
                    score_for_target: RuleSetScore = self._score(
                        rules=tmp_extended_rule_set, training_data=training_data,
                        target_attribute=target_attribute)

                    if score_for_target - current_set_scores[target_attribute] > self.min_required_score_increase:
                        target_scores_per_rule.append((score_for_target, candidate_rule))

                    tmp_extended_rule_set.remove(candidate_rule)

        # find the rule such that when added to the rule set,
        # it increases the rule set score, but limit  not hurt the other scores

        if len(target_scores_per_rule) == 0:  # no rule found
            return None
        else:
            if self.verbose:
                print(
                    f"for target {target_attribute},"
                    f" {len(target_scores_per_rule)} rules lead to an improved prediction")

            # there is at least one rule
            target_scores_per_rule.sort(key=lambda tup: tup[0], reverse=True)
            rules_considered_as_best: List[Tuple[RuleSetScore, MIDSRule]] = self._best_rules_to_consider(
                target_scores_per_rule)

            rule_to_eval_map: Dict[MIDSRule,
                                   Dict[TargetAttr,
                                        RuleSetScore]] = {}
            for score_for_target, mids_rule in rules_considered_as_best:
                extended_rule_set_scores: Dict[TargetAttr, RuleSetScore] = {target_attribute: score_for_target}
                for other_rule_target in mids_rule.get_target_attributes():
                    if other_rule_target != target_attribute:
                        score_for_other_target: RuleSetScore = self._score(rules=tmp_extended_rule_set,
                                                                           training_data=training_data,
                                                                           target_attribute=other_rule_target)
                        extended_rule_set_scores[other_rule_target] = score_for_other_target

                rule_to_eval_map[mids_rule] = extended_rule_set_scores

            best_rule_that_hurts_the_other_targets_the_least: Optional[MIDSRule] = \
                self._get_rule_that_hurts_the_other_targets_the_least(
                    rules_considered_as_best=rules_considered_as_best,
                    rule_to_eval_map=rule_to_eval_map,
                    main_target=target_attribute,
                    current_rule_set_scores=current_set_scores,
                )
            if best_rule_that_hurts_the_other_targets_the_least is None:
                raise Exception("At this point, we should have found a rule, not None")

            if self.verbose:
                old_score = current_set_scores[target_attribute]
                improved_score = rule_to_eval_map[best_rule_that_hurts_the_other_targets_the_least][target_attribute]

                improvement = improved_score - old_score
                print(f"for target {target_attribute}, best rule least hurting others "
                      f"results in score "
                      f"{improved_score:0.4f} (improvement of {improvement:0.4f})")

            best_extended_rule_set = current_set | {best_rule_that_hurts_the_other_targets_the_least}
            scores_best_extended_rule_set: Dict[TargetAttr, RuleSetScore] = self._update_rule_set_scores(
                current_set_scores,
                rule_to_eval_map[best_rule_that_hurts_the_other_targets_the_least]
            )

            possibly_changed_targets = list(rule_to_eval_map[best_rule_that_hurts_the_other_targets_the_least])

            return best_extended_rule_set, scores_best_extended_rule_set, possibly_changed_targets

    @staticmethod
    def _update_rule_set_scores(current_rule_set_scores: Dict[TargetAttr, RuleSetScore],
                                updated_scores: Dict[TargetAttr, RuleSetScore]
                                ) -> Dict[TargetAttr, RuleSetScore]:
        new_rule_set_scores = current_rule_set_scores.copy()
        for target, score in updated_scores.items():
            new_rule_set_scores[target] = score
        return new_rule_set_scores

    @staticmethod
    def close_enough(best_score, other_score, boundary_value: float = 0.1):
        return 0 <= best_score - other_score <= boundary_value

    def _best_rules_to_consider(self,
                                sorted_target_scores_per_rule: List[Tuple[RuleSetScore, MIDSRule]]
                                ) -> List[Tuple[RuleSetScore, MIDSRule]]:
        """
        Given a list of rules sorted on a goodness criterion, take the best rules.
        Best could be defined as:
            the x highest scoring rules
            the rules within a certain percentage of the best rule

        :param sorted_target_scores_per_rule:
        :return:
        """

        best_tuple: Tuple[RuleSetScore, MIDSRule] = sorted_target_scores_per_rule[0]
        score_of_best_rule = best_tuple[0]
        best_rule_for_target = best_tuple[1]

        if self.verbose:
            print(f"best rule results in score {score_of_best_rule}")

        best_rules = [best_tuple]
        for other_tuple in sorted_target_scores_per_rule[1:]:
            score_other_rule = other_tuple[0]
            if self.close_enough(score_of_best_rule, score_other_rule, self.max_score_diff_with_best_rule):
                # other_rule = other_tuple[1]
                best_rules.append(other_tuple)

        if self.verbose:
            print(f"{len(best_rules)} rules are being considered as best...")
        return best_rules

    def _get_rule_that_hurts_the_other_targets_the_least(self,
                                                         rules_considered_as_best: List[Tuple[RuleSetScore, MIDSRule]],
                                                         rule_to_eval_map: Dict[MIDSRule,
                                                                                Dict[TargetAttr,
                                                                                     RuleSetScore]],
                                                         main_target: TargetAttr,
                                                         current_rule_set_scores: Dict[TargetAttr, RuleSetScore]
                                                         ) -> Optional[MIDSRule]:
        """
        We want to limit the damage done to the other targets.

        For each rule:
            for each target:
                calculate the difference in score for this target:
                    score_diff = new_score - old_score.
                    if score_diff > 0  --> good: the rule increases the score on this target
                    if score_diff < 0 --> bad: the rule decreases the score on this target

            min_score_diff = the minimal score_diff over the targets of the rule
                --> the largest damage the rule can do

        we want the rule with the highest min_score diff

        :param rules_considered_as_best:
        :param rule_to_eval_map:
        :param main_target:
        :param current_rule_set_scores:
        :return:
        """

        best_rule: Optional[MIDSRule] = None
        min_score_diff_of_best_rule = float('-inf')

        for rule_set_score, rule in rules_considered_as_best:
            extended_rule_set_scores: Dict[TargetAttr, RuleSetScore] = rule_to_eval_map[rule]

            score_diffs_per_target = []
            for other_rule_target in extended_rule_set_scores.keys():
                if other_rule_target is not main_target:
                    # print(f"\tconsidering score for {other_rule_target}")
                    other_target_score = extended_rule_set_scores[other_rule_target]
                    current_rule_set_score = current_rule_set_scores[other_rule_target]
                    score_diff = other_target_score - current_rule_set_score
                    if self.verbose:
                        print(f"\tconsidering score for {other_rule_target}: {score_diff}")
                    score_diffs_per_target.append(score_diff)
            if len(score_diffs_per_target) == 0:
                min_score_diff = 0
            else:
                min_score_diff = min(score_diffs_per_target)

            if min_score_diff > min_score_diff_of_best_rule:
                best_rule = rule
                min_score_diff_of_best_rule = min_score_diff

        return best_rule

    def fit(self,
            training_data: pd.DataFrame,
            ground_set: Set[MIDSRule],
            rule_combination_strategy=RuleCombiningStrategy.WEIGHTED_VOTE,
            ) -> Tuple[Set[MIDSRule],
                       Dict[TargetAttr,
                            RuleSetScore]]:

        if not isinstance(ground_set, set):
            ground_set = set(ground_set)
        self.ground_set_size = len(ground_set)

        self.rule_combinator: RuleCombinator = rule_combination_strategy.create()
        self.rule_combination_strategy: RuleCombiningStrategy = rule_combination_strategy

        self._set_all_targets_must_be_checked()

        type_check_dataframe(training_data)
        if self.filter_nans and self.should_cache_nan_filtered_training_data:
            self._prepare_cached_training_data(training_data)

        start_time_s = time.time()
        self.default_predictions = get_majority_classes_over_whole_training_data(
            self.target_attributes, training_data
        )

        selected_set: Set[MIDSRule] = set()
        selected_set_scores: Dict[TargetAttr, RuleSetScore] = {}
        for target_attr in self.target_attributes:
            if self.verbose:
                print(f"Calculating init score for {target_attr}...")
            selected_set_scores[target_attr] = self._score(
                rules=selected_set, training_data=training_data, target_attribute=target_attr)
        init_set_scores: Dict[TargetAttr, RuleSetScore] = selected_set_scores.copy()
        if self.keep_score_evolution:
            self._init_score_evolution()
            self._add_to_score_evolution(init_set_scores)

        if self.verbose:
            print("=== start rule Learning ===")
            print("Initial target scores (empty set):", selected_set_scores)

        iteration: int = 1
        target_count: int = 1

        while self._is_target_left_to_check():

            # NOTE: each iteration, we focus on a different target
            # curr_target: TargetAttr = self._change_target()
            curr_target: TargetAttr = self._get_next_target_to_check()
            if self.verbose:
                print(f"It {iteration} - target: {curr_target}  ({target_count}/{len(self.target_attributes)})")

            optional_improvement: Optional[
                Tuple[Set[MIDSRule, Dict, List[TargetAttr]]]] = self.get_best_improvement_for_target(
                training_data=training_data,
                current_set=selected_set,
                current_set_scores=selected_set_scores,
                ground_set=ground_set,
                target_attribute=curr_target,
            )

            if optional_improvement is None:
                if self.verbose:
                    print(f"Found no improvement for {curr_target}")

                self.targets_to_check[curr_target] = False
                target_count += 1
            else:
                improved_set: Set[MIDSRule] = optional_improvement[0]
                scores_of_improved_set: Dict[TargetAttr, RuleSetScore] = optional_improvement[1]
                possibly_changed_targets: List[TargetAttr] = optional_improvement[2]

                selected_set = improved_set
                selected_set_scores = scores_of_improved_set

                if self.keep_score_evolution:
                    self._add_to_score_evolution(scores_of_improved_set)

                if self.verbose:

                    changed_str = ", ".join(possibly_changed_targets)
                    if self.verbose:
                        print(f"resetting improvement found flags for " + changed_str + " ...")
                    for target in possibly_changed_targets:
                        self.targets_to_check[target] = True

                iteration += 1
                target_count = 1
                if self.verbose:
                    print(f"End it. {iteration}")
                    print("-------------------------")

        if self.verbose:
            print("=== end rule learning ===")
            print(f"Found {len(selected_set)} rules")
            print("Score changes from default prediction:")
            for target in self.target_attributes:
                print(f"\t{target}: {init_set_scores[target]:.2f} --> {selected_set_scores[target]:.2f}")
            print("=========================")

        end_time_s = time.time()
        self.learning_time_s = end_time_s - start_time_s
        self.learned_rule_set = selected_set
        self.learned_rule_set_scores = selected_set_scores

        self.nan_filterered_training_data_cache = None

        return selected_set, selected_set_scores

    def predict(self, dataframe: pd.DataFrame, target_attribute: str):
        return self.rule_combinator.predict(
            self.learned_rule_set, dataframe, target_attribute, self.default_predictions[target_attribute])
