from typing import Dict, Tuple, Iterable, Optional

from mdrsl.rule_models.mids.mids_rule import MIDSRule
from mdrsl.rule_models.mids.mids_ruleset import MIDSRuleSet


TargetAttr = str
RuleID = int
Count = int


def estimate_upper_bound_cache_size_in_nb_of_integers(nb_of_rules) -> int:
    """
    Estimate an upper bound for a the size of the cache.

    The cache maps a rule pair sharing at least 1 attribute in their heads to two integer values.
    A key into the cache is also two integers.

    Assume there are n rules.
    If they are ordered,
        rule 1 has to be compared with n-1 rules
        rule 2 with n-2 rules
        ...
        rule n with 0 rules
    --> (n²-n) / 2 possible rule pairs.

    In the absolute worst case, each rule has a shared target attribute with each other rule.
    An upper bound for the nb of integers necessary for the cache is:
        4 * (n²-n) / 2 = 2 (n²-n)
    :param nb_of_rules:
    :return:
    """

    return 2 * (nb_of_rules ** 2 - nb_of_rules)


def create_f2_f3_cache(total_rule_set: MIDSRuleSet, overlap_checker, quant_dataframe,
                       f2_f3_target_attr_to_upper_bound_map: Dict[TargetAttr, int],
                       nb_of_target_attributes: int
                       ):
    """

    create a cache, mapping each (unordered) pair of rules
            to their intra and inter - class overlap counts
    """

    # raise Exception("INCORRECT")
    cache = {}  # type: Dict[Tuple[RuleID,RuleID], Tuple[Count, Count]]

    for i, rule_i in enumerate(total_rule_set.ruleset):
        for j, rule_j in enumerate(total_rule_set.ruleset):
            if i >= j:
                continue

            target_attr_rule_i = rule_i.get_target_attributes()
            target_attr_rule_j = rule_j.get_target_attributes()
            shared_attributes = target_attr_rule_i & target_attr_rule_j

            # if both rules have at least one target attribute in common
            if len(shared_attributes) > 0:
                overlap_count = overlap_checker.get_pure_overlap_count(rule_i, rule_j, quant_dataframe)

                weighed_overlap_count_intra_class_sum = 0
                weighted_overlap_count_inter_class_sum = 0

                for target_attr in shared_attributes:
                    # check whether the rules predict the same value for the target attribute
                    target_value_rule_i = rule_i.get_predicted_value_for(target_attr)
                    target_value_rule_j = rule_j.get_predicted_value_for(target_attr)

                    f2_f3_upper_bound_for_target_attr = f2_f3_target_attr_to_upper_bound_map[target_attr]
                    if target_value_rule_i == target_value_rule_j:
                        weighed_overlap_count_intra_class_sum = weighed_overlap_count_intra_class_sum + \
                            overlap_count / (nb_of_target_attributes * f2_f3_upper_bound_for_target_attr)
                    else:
                        weighted_overlap_count_inter_class_sum = weighted_overlap_count_inter_class_sum + \
                            overlap_count / (nb_of_target_attributes * f2_f3_upper_bound_for_target_attr)

                # weighed_overlap_count_intra_class_sum = weighed_overlap_count_intra_class_sum / nb_of_target_attributes
                # weighted_overlap_count_inter_class_sum = weighted_overlap_count_inter_class_sum / nb_of_target_attributes

                cache_key = get_cache_key(rule_i, rule_j)
                cache[cache_key] = (weighed_overlap_count_intra_class_sum, weighted_overlap_count_inter_class_sum)
    return cache


def f2_f3_combo_minimize_overlap_predicting_the_same_and_different_class_caching(cache: Dict[Tuple[int,int], Tuple[int, int]],
                                                                                 solution_set: MIDSRuleSet
                                                                                 ) -> Tuple[float, float]:
    weighted_overlap_intra_class_sum = 0
    weighted_overlap_inter_class_sum = 0

    for i, rule_i in enumerate(solution_set.ruleset):
        for j, rule_j in enumerate(solution_set.ruleset):
            if i >= j:
                continue

            cached_value = cache.get(get_cache_key(rule_i, rule_j), None)
            if cached_value is not None:
                weighted_overlap_intra_class_sum += cached_value[0]
                weighted_overlap_inter_class_sum += cached_value[1]

    f2 = 1 - weighted_overlap_intra_class_sum
    f3 = 1 - weighted_overlap_inter_class_sum

    # print("f2:", f2)
    # print("f3:", f3)
    if f2 < 0 or f3 < 0:
        print("f2 or f3 less than 0")

    return f2, f3

def f2_f3_value_reuse_minimize_overlap_caching(cache: Dict[Tuple[int, int], Tuple[int, int]],
                                               f2_previous, f3_previous,
                                               rules_intersection_previous_and_current: Iterable[MIDSRule],
                                               added_rules: Iterable[MIDSRule],
                                               deleted_rules: Iterable[MIDSRule]
                                               ) -> Tuple[float, float]:
    f2_weighted_overlap_intra_class_sum_added_rules: int
    f3_weighted_overlap_inter_class_sum_added_rules: int

    f2_weighted_overlap_intra_class_sum_deleted_rules: int
    f3_weighted_overlap_inter_class_sum_deleted_rules: int

    f2_weighted_overlap_intra_class_sum_added_rules, f3_weighted_overlap_inter_class_sum_added_rules = _f2_f3_get_overlap_sum_maps(
        cache, rules_intersection_previous_and_current, added_rules
    )

    f2_weighted_overlap_intra_class_sum_deleted_rules, f3_weighted_overlap_inter_class_sum_deleted_rules = _f2_f3_get_overlap_sum_maps(
        cache, rules_intersection_previous_and_current, deleted_rules
    )

    f2 = f2_previous - f2_weighted_overlap_intra_class_sum_added_rules + f2_weighted_overlap_intra_class_sum_deleted_rules
    f3 = f3_previous - f3_weighted_overlap_inter_class_sum_added_rules + f3_weighted_overlap_inter_class_sum_deleted_rules

    # print("f2:", f2)
    # print("f3:", f3)
    if f2 < 0 or f3 < 0:
        print("f2 or f3 less than 0")

    return f2, f3

TargetAttr = str

def _f2_f3_get_overlap_sum_maps(cache, rules_intersection_previous_and_current: Iterable[MIDSRule],
                                added_or_deleted_rules: Optional[Iterable[MIDSRule]] =None)\
        -> Tuple[int, int]:
    f2_weighted_overlap_intra_class_sum = 0
    f3_weighted_overlap_inter_class_sum = 0

    if added_or_deleted_rules is not None:
        rule_i: MIDSRule
        for rule_i in rules_intersection_previous_and_current:
            rule_a: MIDSRule
            for rule_a in added_or_deleted_rules:

                cached_value = cache.get(get_cache_key(rule_i, rule_a), None)
                if cached_value is not None:
                    f2_weighted_overlap_intra_class_sum += cached_value[0]
                    f3_weighted_overlap_inter_class_sum += cached_value[1]


        rule_i: MIDSRule
        rule_j: MIDSRule
        for i, rule_i in enumerate(added_or_deleted_rules):
            for j, rule_j in enumerate(added_or_deleted_rules):
                if i >= j:
                    continue
                #
                # for i in range(0, len(rules_to_add_or_delete)):
                #     for j in range(i + 1, len(rules_to_add_or_delete)):
                #         rule_i = rules_to_add_or_delete[i]
                #         rule_j = rules_to_add_or_delete[j]

                cached_value = cache.get(get_cache_key(rule_i, rule_j), None)
                if cached_value is not None:
                    f2_weighted_overlap_intra_class_sum += cached_value[0]
                    f3_weighted_overlap_inter_class_sum += cached_value[1]

    return f2_weighted_overlap_intra_class_sum, f3_weighted_overlap_inter_class_sum


def get_cache_key(rule1: MIDSRule, rule2: MIDSRule):
    rule1_id = rule1.get_rule_id()
    rule2_id = rule2.get_rule_id()
    if rule1_id < rule2_id:
        cache_key = (rule1_id, rule2_id)
    elif rule2_id < rule1_id:
        cache_key = (rule2_id, rule1_id)
    else:
        raise Exception("Dont check a rule with itself for overlap")
    return cache_key
