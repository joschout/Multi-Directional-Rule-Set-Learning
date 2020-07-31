from typing import Optional, Dict, List, KeysView

Attr = str


class EncodingBookKeeper:
    """
    Keeps track of a one hot encoding,
    i.e. which original columns map to which set of one-hot-encoded columns,
    and vice versa.
    """

    def __init__(self, ohe_prefix_separator: str):
        self.ohe_prefix_separator = ohe_prefix_separator

        self.__original_to_ohe_attr_map: Optional[
            Dict[Attr, List[Attr]]
        ] = None

        self.__ohe_attr_to_original_map: Optional[
            Dict[Attr, Attr]
        ] = None

    @staticmethod
    def build_encoding_book_keeper_from_ohe_columns(ohe_columns,
                                                    ohe_prefix_separator: str) -> 'EncodingBookKeeper':
        encoding_book_keeper = EncodingBookKeeper(ohe_prefix_separator=ohe_prefix_separator)
        encoding_book_keeper.parse_and_store_one_hot_encoded_columns(ohe_columns=ohe_columns)
        return encoding_book_keeper

    def __str__(self):
        string_representation = f"original_to_ohe:\n"
        for original_attribute, encodings in self.__original_to_ohe_attr_map.items():
            string_representation += f"\t{original_attribute}: {encodings}\n"
        # string_representation += f"ohe_to_original:\n"
        # for encoded_attribute, original_attribute in self.__ohe_attr_to_original_map.items():
        #     string_representation += f"\t{encoded_attribute}: {original_attribute}\n"
        return string_representation

    def set_original_to_ohe_attr_map(self, original_to_ohe_attr_map: Dict[Attr, List[Attr]]) -> None:
        """
        Use this method to initialize an empty EncodingBookKeeper with the given mapping.
        """
        self.__original_to_ohe_attr_map = original_to_ohe_attr_map

        self.__ohe_attr_to_original_map = {}
        for original_attr, ohe_attributes in original_to_ohe_attr_map.items():
            for ohe_attr in ohe_attributes:
                self.__ohe_attr_to_original_map[ohe_attr] = original_attr

    def add_encodings(self, original_to_encoding_map: Dict[Attr, List[Attr]]) -> None:
        """
        Use this method if you want to add extra columns to an already existing mapping.
        """
        if self.__original_to_ohe_attr_map is None:
            self.set_original_to_ohe_attr_map(original_to_encoding_map)
        else:
            for original_attr, ohe_attributes in original_to_encoding_map.items():
                if original_attr in self.get_original_columns():
                    raise Exception("EncodingBookKeeper already contains encodings for " + str(original_attr))
                else:
                    self.__original_to_ohe_attr_map[original_attr] = ohe_attributes
                    for ohe_attr in ohe_attributes:
                        if ohe_attr in self.get_one_hot_encoded_columns():
                            raise Exception("EncodingBookKeeper already contains an encoding with name "
                                            + str(ohe_attr))
                        else:
                            self.__ohe_attr_to_original_map[ohe_attr] = original_attr

    @staticmethod
    def parse_one_hot_encoded_columns(ohe_columns, ohe_prefix_separator) -> Dict[Attr, List[Attr]]:
        """
        Use this static method to parse a list of one-hot encoding columns
            into a mapping of original columns to their one-hot encoding columns.
        """
        original_to_ohe_columns: Dict[Attr, List[Attr]] = {}
        for column in ohe_columns:
            splitted_ohe_column_name = str(column).split(ohe_prefix_separator)
            if len(splitted_ohe_column_name) == 1:
                # don't split it
                original_to_ohe_columns[column] = [column]
            elif len(splitted_ohe_column_name) == 2:
                original_column = splitted_ohe_column_name[0]
                if original_to_ohe_columns.get(original_column, None) is None:
                    original_to_ohe_columns[original_column] = [column]
                else:
                    original_to_ohe_columns[original_column].append(column)
            else:
                raise Exception("Handling split of " + str(column) + " using separator "
                                + ohe_prefix_separator + " failed; got " + str(len(splitted_ohe_column_name))
                                + " piece(s)."
                                )
        return original_to_ohe_columns

    def parse_and_store_one_hot_encoded_columns(self, ohe_columns) -> None:
        """
        Use this method to initialize an empty EncodingBookKeeper by paringing the given one-encoding columns.
        """
        self.set_original_to_ohe_attr_map(
            self.parse_one_hot_encoded_columns(ohe_columns, self.ohe_prefix_separator))

    def get_original_columns(self) -> KeysView[Attr]:
        """
        Get all original columns.
        """
        return self.__original_to_ohe_attr_map.keys()

    def get_one_hot_encoded_columns(self) -> KeysView[Attr]:
        """
        Get all one-hot encoding columns.
        """
        return self.__ohe_attr_to_original_map.keys()

    def get_encodings(self, original_attr: Attr) -> List[Attr]:
        """
        Get the one-hot encoded attributes corresponding to the given original attribute.
        """
        return self.__original_to_ohe_attr_map[original_attr]

    def get_original(self, encoded_attr: Attr) -> Attr:
        """
        Get the original attribute corresponding to the given one-hot encoded attribute.
        """
        return self.__ohe_attr_to_original_map[encoded_attr]
