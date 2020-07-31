import os

from experiments.file_naming.single_target_classifier_indicator import SingleTargetClassifierIndicator
from project_info import project_dir


def assoc_vs_tree_based_single_target_mids_clf_dir() -> str:
    mids_dir: str = os.path.join(project_dir,
                                  'models',
                                  'assoc_vs_trees_single_target_filtered_mids_classifier')
    if not os.path.exists(mids_dir):
        os.makedirs(mids_dir)
    return mids_dir
# --------------------------------------------------------------------------------------------------------------------


def get_single_target_filtered_car_mids_relative_file_name(
        dataset_name: str, fold_i: int, target_attribute: str, confidence_boundary_val: float) -> str:
    return f'{dataset_name}{fold_i}_{target_attribute}_{confidence_boundary_val:0.2f}conf'


def get_single_target_filtered_car_mids_clf_abs_file_name(
        dataset_name: str, fold_i: int,
        target_attribute: str,
        confidence_boundary_val: float) -> str:
    mids_classifier_dir: str = assoc_vs_tree_based_single_target_mids_clf_dir()
    relative_file_name: str = get_single_target_filtered_car_mids_relative_file_name(
        dataset_name=dataset_name, fold_i=fold_i, target_attribute=target_attribute,
        confidence_boundary_val=confidence_boundary_val
    )

    mids_classifier_abs_file_name = os.path.join(mids_classifier_dir, relative_file_name+'_clf.json.gz')
    return mids_classifier_abs_file_name

# --------------------------------------------------------------------------------------------------------------------


def get_single_target_tree_mids_clf_relative_file_name(
        dataset_name: str, fold_i: int,
        target_attribute: str,
        classifier_indicator: SingleTargetClassifierIndicator,
        min_support: float,
        max_depth: int,
        confidence_boundary_val: float
) -> str:
    return (
        f"{dataset_name}{fold_i}_{target_attribute}_{str(classifier_indicator.value)}"
        f"_{min_support}supp_{max_depth}depth"
        f"_{confidence_boundary_val:0.2f}conf"
    )


def get_single_target_filtered_tree_mids_clf_abs_file_name(
        dataset_name: str, fold_i: int,
        target_attribute: str,
        classifier_indicator: SingleTargetClassifierIndicator,
        min_support: float,
        max_depth: int,
        confidence_boundary_val: float
) -> str:

    mids_classifier_dir: str = assoc_vs_tree_based_single_target_mids_clf_dir()
    relative_file_name: str = get_single_target_tree_mids_clf_relative_file_name(
        dataset_name=dataset_name, fold_i=fold_i,
        target_attribute=target_attribute,
        classifier_indicator=classifier_indicator,
        min_support=min_support, max_depth=max_depth,
        confidence_boundary_val=confidence_boundary_val
    )
    mids_classifier_abs_file_name = os.path.join(mids_classifier_dir, relative_file_name + '_clf.json.gz')
    return mids_classifier_abs_file_name

# --------------------------------------------------------------------------------------------------------------------

# pas dit aan

def get_single_target_filtered_tree_mids_target_attr_to_score_info_abs_file_name(
        dataset_name: str, fold_i: int,
        target_attribute: str,
        classifier_indicator: SingleTargetClassifierIndicator,
        min_support: float,
        max_depth: int,
        confidence_boundary_val: float
) -> str:
    mids_classifier_dir: str = assoc_vs_tree_based_single_target_mids_clf_dir()

    relative_file_name: str = get_single_target_tree_mids_clf_relative_file_name(
        dataset_name=dataset_name, fold_i=fold_i,
        target_attribute=target_attribute,
        classifier_indicator=classifier_indicator,
        min_support=min_support, max_depth=max_depth,
        confidence_boundary_val=confidence_boundary_val
    )

    mids_target_attr_to_score_info_abs_file_name = os.path.join(
        mids_classifier_dir, f'{relative_file_name}_target_attr_to_score_info.json.gz')
    return mids_target_attr_to_score_info_abs_file_name


def get_single_target_filtered_tree_mids_interpret_stats_abs_file_name(
        dataset_name: str, fold_i: int,
        target_attribute: str,
        classifier_indicator: SingleTargetClassifierIndicator,
        # nb_of_trees_per_model: int,
        min_support: float,
        max_depth: int,
        confidence_boundary_val: float
    ) -> str:
    mids_classifier_dir: str = assoc_vs_tree_based_single_target_mids_clf_dir()

    relative_file_name: str = get_single_target_tree_mids_clf_relative_file_name(
        dataset_name=dataset_name, fold_i=fold_i,
        target_attribute=target_attribute,
        classifier_indicator=classifier_indicator,
        min_support=min_support, max_depth=max_depth,
        confidence_boundary_val=confidence_boundary_val
    )

    mids_interpret_stats_abs_file_name = os.path.join(
        mids_classifier_dir, f'{relative_file_name}_interpret_stats.json.gz')
    return mids_interpret_stats_abs_file_name


def get_single_target_filtered_car_mids_target_attr_to_score_info_abs_file_name(
        dataset_name: str, fold_i: int,
        target_attribute: str,
        confidence_boundary_val: float
) -> str:
    mids_classifier_dir: str = assoc_vs_tree_based_single_target_mids_clf_dir()

    relative_file_name: str = get_single_target_filtered_car_mids_relative_file_name(
        dataset_name=dataset_name, fold_i=fold_i,
        target_attribute=target_attribute,
        confidence_boundary_val=confidence_boundary_val
    )

    mids_target_attr_to_score_info_abs_file_name = os.path.join(
        mids_classifier_dir, f'{relative_file_name}_target_attr_to_score_info.json.gz')
    return mids_target_attr_to_score_info_abs_file_name


def get_single_target_filtered_car_mids_interpret_stats_abs_file_name(
        dataset_name: str, fold_i: int,
        target_attribute: str,
        confidence_boundary_val: float
    ) -> str:
    mids_classifier_dir: str = assoc_vs_tree_based_single_target_mids_clf_dir()

    relative_file_name: str = get_single_target_filtered_car_mids_relative_file_name(
        dataset_name=dataset_name, fold_i=fold_i,
        target_attribute=target_attribute,
        confidence_boundary_val=confidence_boundary_val
    )

    mids_interpret_stats_abs_file_name = os.path.join(
        mids_classifier_dir, f'{relative_file_name}_interpret_stats.json.gz')
    return mids_interpret_stats_abs_file_name
