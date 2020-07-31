import gzip
import json
from typing import Dict

import jsonpickle


def store_timings_dict(abs_file_name: str, timings_dict: Dict[str, float]) -> None:
    frozen = jsonpickle.encode(timings_dict)
    pretty_frozen = json.dumps(json.loads(frozen), indent=2, sort_keys=True)
    with gzip.open(abs_file_name, 'wt') as ofile:
        ofile.write(pretty_frozen)


def load_timings_dict(abs_file_name: str) -> Dict[str, float]:
    timings_dict: Dict[str, float]
    with gzip.open(abs_file_name, 'rt') as ifile:
        file_contents = ifile.read()
        timings_dict = jsonpickle.decode(file_contents)
    return timings_dict
