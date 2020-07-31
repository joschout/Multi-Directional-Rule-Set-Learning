import os

from experiments.e1_st_association_vs_tree_rules.file_naming.rules.single_target_filtered_cars_naming import \
    assoc_vs_tree_based_single_target_car_dir
from experiments.file_naming.single_target_classifier_indicator import SingleTargetClassifierIndicator


def get_single_target_tree_rules_relative_file_name(
        dataset_name: str, fold_i: int,
        target_attribute: str,
        classifier_indicator: SingleTargetClassifierIndicator,
        min_support: float,
        max_depth: int,
        confidence_boundary_val
) -> str:
    return (
        f"{dataset_name}{fold_i}_{target_attribute}_{str(classifier_indicator.value)}"
        f"_{min_support}supp_{max_depth}depth"
        f"_{confidence_boundary_val:0.2f}conf"
    )


def get_single_target_tree_rules_absolute_file_name(
        dataset_name: str, fold_i: int,
        target_attribute: str,
        classifier_indicator: SingleTargetClassifierIndicator,
        min_support: float,
        max_depth: int,
        confidence_boundary_val
) -> str:
    rules_dir: str = assoc_vs_tree_based_single_target_car_dir()
    relative_file_name: str = get_single_target_tree_rules_relative_file_name(
        dataset_name=dataset_name, fold_i=fold_i,
        target_attribute=target_attribute,
        classifier_indicator=classifier_indicator,
        min_support=min_support, max_depth=max_depth,
        confidence_boundary_val=confidence_boundary_val
    )
    tree_rules_abs_file_name = os.path.join(
        rules_dir,
        f"{relative_file_name}.json.gz"
    )
    return tree_rules_abs_file_name


def get_single_target_tree_rules_gen_timing_info_absolute_file_name(
        dataset_name: str, fold_i: int,
        target_attribute: str,
        classifier_indicator: SingleTargetClassifierIndicator,
        min_support: float,
        max_depth: int,
        confidence_boundary_val
) -> str:
    rules_dir: str = assoc_vs_tree_based_single_target_car_dir()
    relative_file_name: str = get_single_target_tree_rules_relative_file_name(
        dataset_name=dataset_name, fold_i=fold_i,
        target_attribute=target_attribute,
        classifier_indicator=classifier_indicator,
        min_support=min_support, max_depth=max_depth,
        confidence_boundary_val=confidence_boundary_val
    )
    tree_rules_gen_timings_abs_file_name = os.path.join(
        rules_dir,
        f"{relative_file_name}_timings.json.gz"
    )
    return tree_rules_gen_timings_abs_file_name


def get_single_target_random_forest_absolute_file_name(
        dataset_name: str, fold_i: int,
        target_attribute: str,
        classifier_indicator: SingleTargetClassifierIndicator,
        min_support: float,
        max_depth: int,
        confidence_boundary_val
) -> str:
    rules_dir: str = assoc_vs_tree_based_single_target_car_dir()
    relative_file_name: str = get_single_target_tree_rules_relative_file_name(
        dataset_name=dataset_name, fold_i=fold_i,
        target_attribute=target_attribute,
        classifier_indicator=classifier_indicator,
        min_support=min_support, max_depth=max_depth,
        confidence_boundary_val=confidence_boundary_val
    )
    tree_rules_abs_file_name = os.path.join(
        rules_dir,
        f"{relative_file_name}_clf.pickle"
    )
    return tree_rules_abs_file_name
