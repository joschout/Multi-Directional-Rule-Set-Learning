import os

from project_info import project_dir


def assoc_vs_tree_based_single_target_car_dir() -> str:
    mcars_dir: str = os.path.join(project_dir,
                                  'models',
                                  'assoc_vs_trees_single_target_cars')
    if not os.path.exists(mcars_dir):
        os.makedirs(mcars_dir)
    return mcars_dir


def get_single_target_filtered_cars_relative_file_name(
        dataset_name: str, fold_i: int, target_attribute: str, confidence_boundary_val: float) -> str:
    return f'{dataset_name}{fold_i}_{target_attribute}_{confidence_boundary_val:0.2f}conf'


def get_single_target_filtered_cars_abs_filename(
        dataset_name: str, fold_i: int,
        target_attribute: str,
        confidence_boundary_val: float
):
    rules_dir = assoc_vs_tree_based_single_target_car_dir()
    relative_file_name: str = get_single_target_filtered_cars_relative_file_name(
        dataset_name=dataset_name, fold_i=fold_i,
        target_attribute=target_attribute,
        confidence_boundary_val=confidence_boundary_val
    )

    single_target_cars_abs_file_name = os.path.join(rules_dir, f"{relative_file_name}.json.gz")
    return single_target_cars_abs_file_name


def get_single_target_filtered_cars_mining_timings_abs_filename(
        dataset_name: str, fold_i: int,
        target_attribute: str,
        confidence_boundary_val: float
):
    rules_dir = assoc_vs_tree_based_single_target_car_dir()
    relative_file_name: str = get_single_target_filtered_cars_relative_file_name(
        dataset_name=dataset_name, fold_i=fold_i,
        target_attribute=target_attribute,
        confidence_boundary_val=confidence_boundary_val
    )

    single_target_cars_abs_file_name = os.path.join(rules_dir, f"{relative_file_name}_timings.json.gz")
    return single_target_cars_abs_file_name

