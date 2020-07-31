import os

from experiments.file_naming.single_target_classifier_indicator import SingleTargetClassifierIndicator
from project_info import project_dir


def get_merged_single_target_mids_clf_dir() -> str:
    mids_clf_dir: str = os.path.join(project_dir,
                                     'models',
                                     'assoc_vs_trees_merged_single_target_mids_classifier')
    if not os.path.exists(mids_clf_dir):
        os.makedirs(mids_clf_dir)
    return mids_clf_dir


def get_merged_single_target_tree_mids_relative_file_name_without_extension(
        dataset_name: str, fold_i: int,
        classifier_indicator: SingleTargetClassifierIndicator,
        nb_of_trees_per_model: int,
        min_support: float,
        max_depth: int
) -> str:
    return (
        f"{dataset_name}{fold_i}_{str(classifier_indicator.value)}"
        f"_{nb_of_trees_per_model}trees"
        f"_{min_support}supp_{max_depth}depth_merged"
    )


def get_merged_single_target_tree_mids_clf_abs_file_name(
        dataset_name: str, fold_i: int,
        classifier_indicator: SingleTargetClassifierIndicator,
        nb_of_trees_per_model: int,
        min_support: float,
        max_depth: int
) -> str:

    mids_clf_dir: str = get_merged_single_target_mids_clf_dir()
    relative_file_name: str = get_merged_single_target_tree_mids_relative_file_name_without_extension(
        dataset_name=dataset_name, fold_i=fold_i,
        classifier_indicator=classifier_indicator,
        nb_of_trees_per_model=nb_of_trees_per_model,
        min_support=min_support, max_depth=max_depth
    )
    mids_classifier_abs_file_name = os.path.join(
        mids_clf_dir,
        f"{relative_file_name}_clf.json.gz"
    )
    return mids_classifier_abs_file_name
