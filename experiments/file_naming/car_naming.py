import os

from experiments.decision_tree_rule_learning.relative_file_naming import \
    get_tree_derived_rules_rel_file_name_without_extension
from experiments.file_naming.single_target_classifier_indicator import SingleTargetClassifierIndicator

from project_info import project_dir


# --- single target class association rules ----------------------------------------------------------------------------

def get_single_target_cars_dir() -> str:
    cars_dir: str = os.path.join(project_dir, 'models/single_target_cars')
    if not os.path.exists(cars_dir):
        os.makedirs(cars_dir)
    return cars_dir


def get_single_target_cars_dir_for_dataset_fold(dataset_name: str, fold_i: int) -> str:
    cars_dataset_fold_dir: str = os.path.join(get_single_target_cars_dir(), f'{dataset_name}{fold_i}')
    if not os.path.exists(cars_dataset_fold_dir):
        os.makedirs(cars_dataset_fold_dir)
    return cars_dataset_fold_dir


def get_single_target_cars_abs_file_name(dataset_name: str, fold_i: int, target_attribute: str) -> str:
    cars_dataset_fold_dir: str = get_single_target_cars_dir_for_dataset_fold(dataset_name, fold_i)
    cars_abs_file_name = os.path.join(cars_dataset_fold_dir, f'{target_attribute}.json.gz')
    return cars_abs_file_name


def get_single_target_car_mining_timings_abs_file_name(dataset_name: str, fold_i: int, target_attribute: str) -> str:
    cars_dataset_fold_dir: str = get_single_target_cars_dir_for_dataset_fold(dataset_name, fold_i)
    cars_timings_abs_file_name = os.path.join(cars_dataset_fold_dir, f'{target_attribute}_timings.json.gz')
    return cars_timings_abs_file_name

# --- multi target class association rules ----------------------------------------------------------------------------

def get_multi_target_cars_dir() -> str:
    mcars_dir: str = os.path.join(project_dir, 'models/multi_target_cars')
    if not os.path.exists(mcars_dir):
        os.makedirs(mcars_dir)
    return mcars_dir


def get_multi_target_cars_abs_file_name(dataset_name: str, fold_i: int) -> str:
    mcars_dir: str = get_multi_target_cars_dir()
    mcars_abs_file_name = os.path.join(mcars_dir, f'{dataset_name}{fold_i}.json.gz')
    return mcars_abs_file_name


def get_multi_target_car_mining_timings_abs_file_name(dataset_name: str, fold_i: int) -> str:
    mcars_dir: str = get_multi_target_cars_dir()
    mcars_timings_abs_file_name = os.path.join(mcars_dir, f'{dataset_name}{fold_i}_timings.json.gz')
    return mcars_timings_abs_file_name


# --- multi target rules derived from tree classifiers-----------------------------------------------------------------

def get_tree_derived_rules_dir() -> str:
    mcars_dir: str = os.path.join(project_dir, 'models/tree_derived_rules')
    if not os.path.exists(mcars_dir):
        os.makedirs(mcars_dir)
    return mcars_dir


def get_tree_derived_rules_abs_file_name(dataset_name: str, fold_i: int,
                                         classifier_indicator: SingleTargetClassifierIndicator,
                                         nb_of_trees_per_model: int,
                                         nb_of_original_targets_to_predict: int,
                                         min_support: float,
                                         max_depth: int) -> str:
    rules_dir: str = get_tree_derived_rules_dir()
    relative_file_name = get_tree_derived_rules_rel_file_name_without_extension(
            dataset_name=dataset_name, fold_i=fold_i,
            classifier_indicator=classifier_indicator, nb_of_trees_per_model=nb_of_trees_per_model,
            nb_of_original_targets_to_predict=nb_of_original_targets_to_predict,
            min_support=min_support, max_depth=max_depth
            )
    tree_derived_rule_abs_file_name = os.path.join(rules_dir, f"{relative_file_name}.json.gz")
    return tree_derived_rule_abs_file_name


def get_tree_derived_rules_gen_timing_info_abs_file_name(dataset_name: str, fold_i: int,
                                         classifier_indicator: SingleTargetClassifierIndicator,
                                         nb_of_trees_per_model: int,
                                         nb_of_original_targets_to_predict: int,
                                         min_support: float,
                                         max_depth: int) -> str:
    rules_dir: str = get_tree_derived_rules_dir()
    relative_file_name = get_tree_derived_rules_rel_file_name_without_extension(
            dataset_name=dataset_name, fold_i=fold_i,
            classifier_indicator=classifier_indicator, nb_of_trees_per_model=nb_of_trees_per_model,
            nb_of_original_targets_to_predict=nb_of_original_targets_to_predict,
            min_support=min_support, max_depth=max_depth
            )
    tree_derived_rule_abs_file_name = os.path.join(rules_dir, f"{relative_file_name}_timings.json.gz")
    return tree_derived_rule_abs_file_name


def get_tree_derived_rules_logger_abs_file_name(
        dataset_name: str, fold_i: int,
        classifier_indicator: SingleTargetClassifierIndicator,
        nb_of_trees_per_model: int,
        nb_of_original_targets_to_predict: int,
        min_support: float,
        max_depth: int) -> str:
    rules_dir: str = get_tree_derived_rules_dir()
    relative_file_name = get_tree_derived_rules_rel_file_name_without_extension(
            dataset_name=dataset_name, fold_i=fold_i,
            classifier_indicator=classifier_indicator, nb_of_trees_per_model=nb_of_trees_per_model,
            nb_of_original_targets_to_predict=nb_of_original_targets_to_predict,
            min_support=min_support, max_depth=max_depth
            )
    tree_derived_rule_abs_file_name = os.path.join(rules_dir, f"{relative_file_name}.log")
    return tree_derived_rule_abs_file_name
