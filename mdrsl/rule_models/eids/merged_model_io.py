import gzip

import jsonpickle

from mdrsl.rule_models.eids.st_to_mt_model_merging import MergedSTMIDSClassifier


def store_merged_st_mids_model(merged_model_abs_file_name: str, merged_st_mids_classifier: MergedSTMIDSClassifier) -> None:
    frozen = jsonpickle.encode(merged_st_mids_classifier)
    with gzip.open(merged_model_abs_file_name, 'wt') as ofile:
        ofile.write(frozen)


def load_merged_st_mids_model(merged_model_abs_file_name: str) -> MergedSTMIDSClassifier:
    mids_classifier: MergedSTMIDSClassifier
    with gzip.open(merged_model_abs_file_name, 'rt') as ifile:
        file_contents = ifile.read()
        mids_classifier = jsonpickle.decode(file_contents)
    return mids_classifier
