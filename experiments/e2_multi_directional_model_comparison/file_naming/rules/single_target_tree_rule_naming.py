import os

from experiments.file_naming.single_target_classifier_indicator import SingleTargetClassifierIndicator
from project_info import project_dir


def get_single_target_tree_rule_dir() -> str:
    mcars_dir: str = os.path.join(project_dir,
                                  'models',
                                  'single_target_tree_rules')
    if not os.path.exists(mcars_dir):
        os.makedirs(mcars_dir)
    return mcars_dir


def get_single_target_tree_rules_relative_file_name_without_extension(
        dataset_name: str, fold_i: int,
        target_attribute: str,
        classifier_indicator: SingleTargetClassifierIndicator,
        nb_of_trees_per_model: int,
        min_support: float,
        max_depth: int
) -> str:
    return (
        f"{dataset_name}{fold_i}_{target_attribute}_{str(classifier_indicator.value)}"
        f"_{nb_of_trees_per_model}trees"
        f"_{min_support}supp_{max_depth}depth"
    )


def get_single_target_tree_rules_abs_file_name(
        dataset_name: str, fold_i: int,
        target_attribute: str,
        classifier_indicator: SingleTargetClassifierIndicator,
        nb_of_trees_per_model: int,
        min_support: float,
        max_depth: int,
):
    rules_dir = get_single_target_tree_rule_dir()
    relative_file_name: str = get_single_target_tree_rules_relative_file_name_without_extension(
        dataset_name=dataset_name, fold_i=fold_i,
        target_attribute=target_attribute,
        classifier_indicator=classifier_indicator,
        nb_of_trees_per_model=nb_of_trees_per_model,
        min_support=min_support, max_depth=max_depth
    )

    tree_derived_rule_abs_file_name = os.path.join(rules_dir, f"{relative_file_name}.json.gz")
    return tree_derived_rule_abs_file_name


def get_single_target_tree_rules_gen_timing_info_abs_file_name(
        dataset_name: str, fold_i: int,
        target_attribute: str,
        classifier_indicator: SingleTargetClassifierIndicator,
        nb_of_trees_per_model: int,
        min_support: float,
        max_depth: int,
):
    rules_dir = get_single_target_tree_rule_dir()
    relative_file_name: str = get_single_target_tree_rules_relative_file_name_without_extension(
        dataset_name=dataset_name, fold_i=fold_i,
        target_attribute=target_attribute,
        classifier_indicator=classifier_indicator,
        nb_of_trees_per_model=nb_of_trees_per_model,
        min_support=min_support, max_depth=max_depth
    )

    tree_derived_rule_abs_file_name = os.path.join(rules_dir, f"{relative_file_name}_timings.json.gz")
    return tree_derived_rule_abs_file_name
