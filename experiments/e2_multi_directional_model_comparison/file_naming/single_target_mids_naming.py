import os

from experiments.e2_multi_directional_model_comparison.file_naming.rules.single_target_tree_rule_naming import \
    get_single_target_tree_rules_relative_file_name_without_extension
from experiments.file_naming.single_target_classifier_indicator import SingleTargetClassifierIndicator

from project_info import project_dir


def get_single_target_mids_clf_dir() -> str:
    mids_clf_dir: str = os.path.join(project_dir,
                                     'models',
                                     'assoc_vs_trees_single_target_mids_classifier')
    if not os.path.exists(mids_clf_dir):
        os.makedirs(mids_clf_dir)
    return mids_clf_dir


def get_single_target_tree_mids_clf_abs_file_name(
        dataset_name: str, fold_i: int,
        target_attribute: str,
        classifier_indicator: SingleTargetClassifierIndicator,
        nb_of_trees_per_model: int,
        min_support: float,
        max_depth: int
) -> str:

    mids_classifier_dir: str = get_single_target_mids_clf_dir()
    relative_file_name: str = get_single_target_tree_rules_relative_file_name_without_extension(
        dataset_name=dataset_name, fold_i=fold_i,
        target_attribute=target_attribute,
        classifier_indicator=classifier_indicator,
        nb_of_trees_per_model=nb_of_trees_per_model,
        min_support=min_support, max_depth=max_depth,
    )
    mids_classifier_abs_file_name = os.path.join(mids_classifier_dir, relative_file_name + '_clf.json.gz')
    return mids_classifier_abs_file_name


def get_single_target_car_mids_clf_relative_file_name(
        dataset_name: str, fold_i: int, target_attribute: str) -> str:
    return f'{dataset_name}{fold_i}_{target_attribute}'


def get_single_target_car_mids_clf_abs_file_name(dataset_name: str, fold_i: int,
                                                 target_attribute: str) -> str:
    mids_classifier_dir: str = get_single_target_mids_clf_dir()
    relative_file_name: str = get_single_target_car_mids_clf_relative_file_name(dataset_name, fold_i, target_attribute)
    mids_classifier_abs_file_name = os.path.join(mids_classifier_dir, f'{relative_file_name}_clf.json.gz')
    return mids_classifier_abs_file_name


def get_single_target_tree_mids_target_attr_to_score_info_abs_file_name(
        dataset_name: str, fold_i: int,
        target_attribute: str,
        classifier_indicator: SingleTargetClassifierIndicator,
        nb_of_trees_per_model: int,
        min_support: float,
        max_depth: int
) -> str:
    mids_classifier_dir: str = get_single_target_mids_clf_dir()

    relative_file_name: str = get_single_target_tree_rules_relative_file_name_without_extension(
        dataset_name=dataset_name, fold_i=fold_i,
        target_attribute=target_attribute,
        classifier_indicator=classifier_indicator,
        nb_of_trees_per_model=nb_of_trees_per_model,
        min_support=min_support, max_depth=max_depth,
    )

    mids_target_attr_to_score_info_abs_file_name = os.path.join(
        mids_classifier_dir, f'{relative_file_name}_target_attr_to_score_info.json.gz')
    return mids_target_attr_to_score_info_abs_file_name


def get_single_target_tree_mids_interpret_stats_abs_file_name(
        dataset_name: str, fold_i: int,
        target_attribute: str,
        classifier_indicator: SingleTargetClassifierIndicator,
        nb_of_trees_per_model: int,
        min_support: float,
        max_depth: int
    ) -> str:
    mids_classifier_dir: str = get_single_target_mids_clf_dir()

    relative_file_name: str = get_single_target_tree_rules_relative_file_name_without_extension(
        dataset_name=dataset_name, fold_i=fold_i,
        target_attribute=target_attribute,
        classifier_indicator=classifier_indicator,
        nb_of_trees_per_model=nb_of_trees_per_model,
        min_support=min_support, max_depth=max_depth,
    )

    mids_interpret_stats_abs_file_name = os.path.join(
        mids_classifier_dir, f'{relative_file_name}_interpret_stats.json.gz')
    return mids_interpret_stats_abs_file_name


def get_single_target_car_mids_target_attr_to_score_info_abs_file_name(
        dataset_name: str, fold_i: int,
        target_attribute: str,
) -> str:
    mids_classifier_dir: str = get_single_target_mids_clf_dir()

    relative_file_name: str = get_single_target_car_mids_clf_abs_file_name(
        dataset_name=dataset_name, fold_i=fold_i,
        target_attribute=target_attribute,
    )

    mids_target_attr_to_score_info_abs_file_name = os.path.join(
        mids_classifier_dir, f'{relative_file_name}_target_attr_to_score_info.json.gz')
    return mids_target_attr_to_score_info_abs_file_name


def get_single_target_car_mids_interpret_stats_abs_file_name(
        dataset_name: str, fold_i: int,
        target_attribute: str,
    ) -> str:
    mids_classifier_dir: str = get_single_target_mids_clf_dir()

    relative_file_name: str = get_single_target_car_mids_clf_abs_file_name(
        dataset_name=dataset_name, fold_i=fold_i,
        target_attribute=target_attribute,
    )

    mids_interpret_stats_abs_file_name = os.path.join(
        mids_classifier_dir, f'{relative_file_name}_interpret_stats.json.gz')
    return mids_interpret_stats_abs_file_name
