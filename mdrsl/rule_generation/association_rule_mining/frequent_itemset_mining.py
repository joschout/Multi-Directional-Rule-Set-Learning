from typing import List, Optional, Union, Tuple

# import fim
import pandas as pd

from bidict import bidict

attribute_value_separator = ';=;'

filter_missing_values = True

Item = str
ItemEncoding = int
Transaction = Union[List[Item], List[ItemEncoding]]


def attr_val_to_item(attr_name: str, val: object, separator=None) -> Item:
    if separator is None:
        separator = attribute_value_separator
    return attr_name + separator + str(val)


def dataframe_to_list_of_transactions(df: pd.DataFrame) -> List[Transaction]:
    """
    convert the dataframe to a list of transactions, with each transaction of the form
           [ 'col1=val1', 'col2=val2', ...]
           e.g. ['Passenger_Cat=3rd_class', 'Age_Cat=adult', 'Gender=male']
    """

    list_of_transactions: List[Transaction] = []
    row_index: int
    for row_index in range(0, df.shape[0]):
        row_itemset: Transaction = []
        for col_name in df.columns:
            try:
                column_series = df[col_name]
                row_value = column_series.iloc[row_index]
                is_value_na: bool = pd.isna(row_value)
                if (filter_missing_values and not is_value_na)\
                        or not filter_missing_values:
                    item: Item = attr_val_to_item(col_name, row_value)
                    row_itemset.append(item)
            except Exception as err:
                print(f"column {col_name} and index {row_index}")
                raise err
        list_of_transactions.append(row_itemset)
    return list_of_transactions


class ItemEncoder:

    def __init__(self):
        self.encoding_bidict: bidict = bidict()
        self.next_encoding = 1

    def encode_item(self, item: Item) -> int:
        optional_encoding: Optional[int] = self.encoding_bidict.get(item, None)
        if optional_encoding is not None:
            return optional_encoding
        else:
            item_encoding = self.next_encoding
            self.encoding_bidict[item] = item_encoding
            self.next_encoding += 1
            return item_encoding

    def decode_item(self, encoding: int) -> Item:
        optional_item: Optional[Item] = self.encoding_bidict.inverse.get(encoding, None)
        if optional_item is not None:
            return optional_item
        else:
            raise Exception(f"No item for encoding {encoding}")


def dataframe_to_list_of_transactions_with_encoding(df: pd.DataFrame) -> Tuple[List[Transaction], ItemEncoder]:
    """
    convert the dataframe to a list of transactions, with each transaction of the form
           [ 'col1=val1', 'col2=val2', ...]
           e.g. ['Passenger_Cat=3rd_class', 'Age_Cat=adult', 'Gender=male']
    """

    item_encoder = ItemEncoder()

    list_of_transactions: List[Transaction] = []

    for row_index in range(0, df.shape[0]):
        row_itemset: Transaction = []
        for col_name in df.columns:
            try:
                column_series = df[col_name]
                row_value = column_series.iloc[row_index]
                is_value_na: bool = pd.isna(row_value)
                if (filter_missing_values and not is_value_na)\
                        or not filter_missing_values:
                    item: Item = attr_val_to_item(col_name, row_value)
                    item_encoding: int = item_encoder.encode_item(item)
                    row_itemset.append(item_encoding)
            except Exception as err:
                print(f"column {col_name} and index {row_index}")
                raise err
        list_of_transactions.append(row_itemset)
    return list_of_transactions, item_encoder


def run_fim_apriori(df: pd.DataFrame, min_suppport_thr: float) -> List[Transaction]:
    try:
        import fim
    except Exception as e:
        raise e

    print("running fim apriori function")
    dataset_transactions: List[Transaction] = dataframe_to_list_of_transactions(df)
    print("dataset processed")

    frequent_itemsets_raw = fim.apriori(dataset_transactions, supp=(min_suppport_thr*100))  # List[Tuple[]]
    print("apriori runned")

    frequent_itemsets: List[Transaction] = list(map(lambda i: list(i[0]), frequent_itemsets_raw))  # Li
    print("apriori results processed")
    return frequent_itemsets


def run_apyori_apriori(df: pd.DataFrame, min_suppport_thr: float) -> List[Transaction]:
    """
    Takes a data frame and a support threshold and returns itemsets which satisfy the threshold.

    The idea is to basically
     1. make a list of strings out of the df
     2. and run apriori api on it
     3. return the frequent itemsets

    :param df: dataframe, where each row is a viewed as a transaction
    :param min_suppport_thr:
    :return:
    """
    from mdrsl.rule_generation.association_rule_mining.apyori_impl.apyori import RelationRecord, apriori
    from mdrsl.rule_generation.association_rule_mining.apyori_impl.apyori_utils import print_relation_record

    dataset_transactions: List[Transaction] = dataframe_to_list_of_transactions(df)

    results: List[RelationRecord] = list(apriori(dataset_transactions, min_support=min_suppport_thr))

    for relation_record in results:
        print_relation_record(relation_record)
        print("=====================================")

    list_of_frequent_itemsets: List[Transaction] = []
    for relation_record in results:  # type: RelationRecord
        itemset: Transaction = []
        for pred in relation_record.items:
            itemset.append(pred)
        list_of_frequent_itemsets.append(itemset)

    return list_of_frequent_itemsets


def run_apriori(implementation: str, df: pd.DataFrame, min_suppport_thr: float) -> List[Transaction]:
    if implementation == 'apyori':
        return run_apyori_apriori(df, min_suppport_thr)
    elif implementation == 'fim':
        return run_fim_apriori(df, min_suppport_thr)
    else:
        raise NotImplementedError('No Apriori implementation found for' + implementation)
