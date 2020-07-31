import os

from experiments.decision_tree_rule_learning.relative_file_naming import \
    get_tree_derived_rules_rel_file_name_without_extension
from experiments.file_naming.single_target_classifier_indicator import SingleTargetClassifierIndicator

from mdrsl.project_info import project_dir


def get_models_dir() -> str:
    models_dir: str = os.path.join(project_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    return models_dir


def greedy_models_tree_based_dir() -> str:
    tree_based_greedy_clf: str = os.path.join(get_models_dir(), 'greedy_clf_tree_based')
    if not os.path.exists(tree_based_greedy_clf):
        os.makedirs(tree_based_greedy_clf)
    return tree_based_greedy_clf


def get_tree_based_greedy_clf_abs_file_name(
        dataset_name: str, fold_i: int,
        classifier_indicator: SingleTargetClassifierIndicator,
        nb_of_trees_per_model: int,
        nb_of_original_targets_to_predict: int,
        min_support: float,
        max_depth: int
) -> str:
    greedy_clf_dir: str = greedy_models_tree_based_dir()

    relative_file_name = get_tree_derived_rules_rel_file_name_without_extension(
        dataset_name=dataset_name, fold_i=fold_i,
        classifier_indicator=classifier_indicator, nb_of_trees_per_model=nb_of_trees_per_model,
        nb_of_original_targets_to_predict=nb_of_original_targets_to_predict,
        min_support=min_support, max_depth=max_depth
    )
    greedy_clf_abs_file_name = os.path.join(greedy_clf_dir, f'{relative_file_name}_greedy_naive_clf.json.gz')
    return greedy_clf_abs_file_name


def get_tree_based_greedy_clf_target_attr_to_score_info_abs_file_name(
        dataset_name: str, fold_i: int,
        classifier_indicator: SingleTargetClassifierIndicator,
        nb_of_trees_per_model: int,
        nb_of_original_targets_to_predict: int,
        min_support: float,
        max_depth: int
) -> str:
    greedy_clf_dir: str = greedy_models_tree_based_dir()

    relative_file_name = get_tree_derived_rules_rel_file_name_without_extension(
        dataset_name=dataset_name, fold_i=fold_i,
        classifier_indicator=classifier_indicator, nb_of_trees_per_model=nb_of_trees_per_model,
        nb_of_original_targets_to_predict=nb_of_original_targets_to_predict,
        min_support=min_support, max_depth=max_depth
    )

    greedy_clf_target_attr_to_score_info_abs_file_name = os.path.join(
        greedy_clf_dir, f'{relative_file_name}_target_attr_to_score_info.json.gz')
    return greedy_clf_target_attr_to_score_info_abs_file_name


def get_tree_based_greedy_clf_interpret_stats_abs_file_name(
        dataset_name: str, fold_i: int,
        classifier_indicator: SingleTargetClassifierIndicator,
        nb_of_trees_per_model: int,
        nb_of_original_targets_to_predict: int,
        min_support: float,
        max_depth: int
) -> str:
    greedy_clf_dir: str = greedy_models_tree_based_dir()

    relative_file_name = get_tree_derived_rules_rel_file_name_without_extension(
        dataset_name=dataset_name, fold_i=fold_i,
        classifier_indicator=classifier_indicator, nb_of_trees_per_model=nb_of_trees_per_model,
        nb_of_original_targets_to_predict=nb_of_original_targets_to_predict,
        min_support=min_support, max_depth=max_depth
    )

    greedy_clf_interpret_stats_abs_file_name = os.path.join(
        greedy_clf_dir, f'{relative_file_name}_interpret_stats.json.gz')
    return greedy_clf_interpret_stats_abs_file_name
