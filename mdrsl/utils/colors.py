from typing import Tuple


def rgb_int_to_float(rgb_ints: Tuple[int, int, int]) -> Tuple[float, float, float]:
    r_fl: float = rgb_ints[0] / 255.0
    g_fl: float = rgb_ints[1] / 255.0
    b_fl: float = rgb_ints[2] / 255.0
    return r_fl, g_fl, b_fl
