import os

from experiments.decision_tree_rule_learning.relative_file_naming import \
    get_tree_derived_rules_rel_file_name_without_extension
from experiments.file_naming.classifier_naming import get_single_target_classifier_dir_for_dataset_fold, \
    get_mids_dir, get_tree_based_mids_dir
from experiments.file_naming.single_target_classifier_indicator import SingleTargetClassifierIndicator


def get_single_target_classifier_score_info_abs_file_name(classifier_indicator: SingleTargetClassifierIndicator,
                                                          dataset_name: str, fold_i: int, target_attribute: str) -> str:
    dataset_fold_dir: str = get_single_target_classifier_dir_for_dataset_fold(classifier_indicator,
                                                                              dataset_name, fold_i)
    score_info_abs_file_name: str = os.path.join(dataset_fold_dir, f'{target_attribute}_score_info.json.gz')
    return score_info_abs_file_name


def get_single_target_classifier_interpret_stats_abs_file_name(classifier_indicator: SingleTargetClassifierIndicator,
                                                               dataset_name: str, fold_i: int,
                                                               target_attribute: str) -> str:
    dataset_fold_dir: str = get_single_target_classifier_dir_for_dataset_fold(classifier_indicator,
                                                                              dataset_name, fold_i)
    interpret_stats_abs_file_name: str = os.path.join(
        dataset_fold_dir, f'{target_attribute}_interpret_stats.json.gz')
    return interpret_stats_abs_file_name


# --- MIDS ------------------------------------------------------------------------------------------------------------

def get_mids_target_attr_to_score_info_abs_file_name(dataset_name: str, fold_i: int) -> str:
    mids_classifier_dir: str = get_mids_dir()
    mids_target_attr_to_score_info_abs_file_name = os.path.join(
        mids_classifier_dir, f'{dataset_name}{fold_i}_target_attr_to_score_info.json.gz')
    return mids_target_attr_to_score_info_abs_file_name


def get_mids_target_attr_to_score_info_weighted_voting_abs_file_name(dataset_name: str, fold_i: int) -> str:
    mids_classifier_dir: str = get_mids_dir()
    mids_target_attr_to_score_info_abs_file_name = os.path.join(
        mids_classifier_dir, f'{dataset_name}{fold_i}_target_attr_to_score_info_weighted_voting.json.gz')
    return mids_target_attr_to_score_info_abs_file_name


def get_mids_target_attr_to_score_info_f1_based_abs_file_name(dataset_name: str, fold_i: int) -> str:
    mids_classifier_dir: str = get_mids_dir()
    mids_target_attr_to_score_info_abs_file_name = os.path.join(
        mids_classifier_dir, f'{dataset_name}{fold_i}_target_attr_to_score_info_f1_based.json.gz')
    return mids_target_attr_to_score_info_abs_file_name


def get_mids_interpret_stats_abs_file_name(dataset_name: str, fold_i: int) -> str:
    mids_classifier_dir: str = get_mids_dir()
    mids_interpret_stats_abs_file_name = os.path.join(
        mids_classifier_dir, f'{dataset_name}{fold_i}_interpret_stats.json.gz')
    return mids_interpret_stats_abs_file_name


# --- tree-based MIDS -------------------------------------------------------------------------------------------------

def get_tree_based_mids_target_attr_to_score_info_abs_file_name(
        dataset_name: str, fold_i: int,
        classifier_indicator: SingleTargetClassifierIndicator,
        nb_of_trees_per_model: int,
        nb_of_original_targets_to_predict: int,
        min_support: float,
        max_depth: int
) -> str:
    mids_classifier_dir: str = get_tree_based_mids_dir()

    relative_file_name = get_tree_derived_rules_rel_file_name_without_extension(
            dataset_name=dataset_name, fold_i=fold_i,
            classifier_indicator=classifier_indicator, nb_of_trees_per_model=nb_of_trees_per_model,
            nb_of_original_targets_to_predict=nb_of_original_targets_to_predict,
            min_support=min_support, max_depth=max_depth
            )

    mids_target_attr_to_score_info_abs_file_name = os.path.join(
        mids_classifier_dir, f'{relative_file_name}_target_attr_to_score_info.json.gz')
    return mids_target_attr_to_score_info_abs_file_name


def get_tree_based_mids_interpret_stats_abs_file_name(dataset_name: str, fold_i: int,
                                                      classifier_indicator: SingleTargetClassifierIndicator,
                                                      nb_of_trees_per_model: int,
                                                      nb_of_original_targets_to_predict: int,
                                                      min_support: float,
                                                      max_depth: int
                                                      ) -> str:
    mids_classifier_dir: str = get_tree_based_mids_dir()

    relative_file_name = get_tree_derived_rules_rel_file_name_without_extension(
            dataset_name=dataset_name, fold_i=fold_i,
            classifier_indicator=classifier_indicator, nb_of_trees_per_model=nb_of_trees_per_model,
            nb_of_original_targets_to_predict=nb_of_original_targets_to_predict,
            min_support=min_support, max_depth=max_depth
            )

    mids_interpret_stats_abs_file_name = os.path.join(
        mids_classifier_dir, f'{relative_file_name}_interpret_stats.json.gz')
    return mids_interpret_stats_abs_file_name
