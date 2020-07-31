import os
import time
from typing import Optional

TargetAttr = str


def check_file_last_modified_duration_in_hours(abs_file_name: str ) -> Optional[float]:
    if os.path.isfile(abs_file_name):
        time_point_last_modified_s: float = os.path.getmtime(abs_file_name)
        current_time_s: float = time.time()
        time_range_since_last_modified_s = current_time_s - time_point_last_modified_s
        minutes = time_range_since_last_modified_s / 60.0  # 120 minutes
        hours = minutes / 60  # 2 hours
        return hours
    else:
        return None


def file_does_not_exist_or_has_been_created_earlier_than_(abs_file_name, nb_hours):
    if not os.path.isfile(abs_file_name):
        return True  # file does not exists
    else:
        hours_since_last_modified: float = check_file_last_modified_duration_in_hours(
            abs_file_name
        )
        if hours_since_last_modified is None:
            raise Exception(f"Sudden disappearance of file {abs_file_name}")
        elif hours_since_last_modified > nb_hours:
            return True  # file is old enough
    return False  # File exists and but is not old enough
