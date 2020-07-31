import gzip
import json

import jsonpickle

from mdrsl.data_handling.one_hot_encoding.encoding_book_keeping import EncodingBookKeeper


def store_encoding_book_keeper(abs_file_name: str, encoding_book_keeper: EncodingBookKeeper) -> None:
    frozen = jsonpickle.encode(encoding_book_keeper)
    pretty_frozen = json.dumps(json.loads(frozen), indent=2, sort_keys=True)
    with gzip.open(abs_file_name, 'wt') as ofile:
        ofile.write(pretty_frozen)


def load_encoding_book_keeper(abs_file_name: str) -> EncodingBookKeeper:
    encoding_book_keeper: EncodingBookKeeper
    with gzip.open(abs_file_name, 'rt') as ifile:
        file_contents = ifile.read()
        encoding_book_keeper = jsonpickle.decode(file_contents)
    return encoding_book_keeper
