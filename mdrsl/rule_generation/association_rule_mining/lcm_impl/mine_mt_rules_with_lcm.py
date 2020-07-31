import pickle
import subprocess
import time
from typing import List, Optional, Tuple, Dict

import pandas as pd

from mdrsl.data_structures.rules.multi_target_class_association_rule import MCAR
from mdrsl.data_structures.rules.rule_part import Consequent
from mdrsl.data_structures.rules.generalized_rule_part import GeneralizedAntecedent
from mdrsl.data_structures.item import EQLiteral
from mdrsl.rule_generation.association_rule_mining.frequent_itemset_mining import (
    ItemEncoder, Transaction, dataframe_to_list_of_transactions_with_encoding, attribute_value_separator)

from mdrsl.rule_generation.association_rule_mining.apyori_impl.apyori import (
    TransactionManager, SupportRecord, OrderedStatistic, gen_ordered_statistics, RelationRecord,
    filter_ordered_statistics)


def convert_dataset_to_transaction_file(
        dataframe: pd.DataFrame,
        # dataset_name: str,
        # fold_i: int,
        encoded_transactions_abs_file_name: str,
        item_encoder_abs_file_name: str
        # support: float,
) -> Tuple[List[Transaction], ItemEncoder]:
    # load the required training data of the dataset fold.
    # original_train_data_fold_abs_file_name = get_original_data_fold_abs_file_name(dataset_name, fold_i,
    #                                                                               TrainTestEnum.train)
    # df_train_original_column_order = pd.read_csv(original_train_data_fold_abs_file_name,
    #                                              delimiter=',')

    transactions_list: List[Transaction]
    item_encoder: ItemEncoder
    encoded_transactions_list, item_encoder = dataframe_to_list_of_transactions_with_encoding(dataframe)

    # write_out
    # transactions_output_file_abs_file_name: str = f'tmp/{dataset_name}{fold_i}_transaction_ids.txt'
    # item_encoder_abs_file_name: str = f'tmp/{dataset_name}{fold_i}_item_encoder.json.gz'

    with open(encoded_transactions_abs_file_name, 'w') as transactions_output_file:
        for transact in encoded_transactions_list:
            transactions_output_file.write(
                " ".join([str(item) for item in transact])
            )
            transactions_output_file.write('\n')

    with open(item_encoder_abs_file_name, 'wb') as ofile:
        pickle.dump(item_encoder, ofile)
    # frozen = jsonpickle.encode(item_encoder)
    # with gzip.open(item_encoder_abs_file_name, 'wt') as item_encoder_output_file:
    #     item_encoder_output_file.write(frozen)
    return encoded_transactions_list, item_encoder


def load_encoded_transactions(encoded_transactions_abs_file_name: str, ) -> List[Transaction]:
    transactions_list: List[Transaction] = []

    with open(encoded_transactions_abs_file_name, 'r') as transactions_file:
        for transaction_line in transactions_file:
            transaction_line_parts = transaction_line[:-1].split(" ")
            transactions_items = [int(item_str) for item_str in transaction_line_parts]
            transactions_list.append(transactions_items)

    return transactions_list


def load_item_encoder(item_encoder_abs_file_name) -> ItemEncoder:
    item_encoder: ItemEncoder
    with open(item_encoder_abs_file_name, 'rb') as ifile:
        item_encoder = pickle.load(ifile)
    return item_encoder


def lcm_frequent_itemset_mining(
        min_support,
        max_length: Optional[int],
        total_nb_of_transactions: int,
        encoded_transactions_abs_file_name: str,
        encoded_frequent_itemsets_abs_file_name: str,
        lcm_command: str = None,
):
    command_str = f"Ff"

    if max_length is not None:
        options_str = f"-u {max_length}"
    else:
        options_str = ""

    support_as_int = int(min_support * total_nb_of_transactions)
    if lcm_command is None:
        lcm_command = "lcm"

    lcm_full_command = f"{lcm_command} {command_str} {options_str} {encoded_transactions_abs_file_name} " \
                       f"{support_as_int} {encoded_frequent_itemsets_abs_file_name}"

    return lcm_full_command


def line_to_total_nb_of_transactions(line: str) -> int:
    # remove ending character
    line = line[:-1]
    line = line.replace(" ", "")
    if not line.startswith("(") or not line.endswith(")"):
        raise Exception(f"Wrong first line format: {line}")
    return int(line[1:-1])


def line_to_frequent_itemset(line: str, total_nb_of_transactions: int) -> SupportRecord:
    # remove ending character
    line = line[:-1]

    parts = line.split(" ")

    items_as_id_strs = parts[:-1]
    support_as_str = parts[-1]
    if not support_as_str.startswith("(") or not support_as_str.endswith(")"):
        raise Exception(f"Wrong support format: {support_as_str}")
    support_as_str = support_as_str[1:-1]
    support_as_integer = int(support_as_str)
    support = float(support_as_integer) / float(total_nb_of_transactions)

    item_ids = [int(item_str) for item_str in items_as_id_strs]

    item_set = frozenset(item_ids)
    return SupportRecord(item_set, support)


def read_in_frequent_itemsets(encoded_frequent_itemsets_abs_file_name: str):
    with open(encoded_frequent_itemsets_abs_file_name, "r") as frequent_itemset_file:
        # first line == nb of tran
        first_line = next(frequent_itemset_file)
        total_nb_of_transactions: int = line_to_total_nb_of_transactions(first_line)

        support_records = []
        for line in frequent_itemset_file:
            fi_support_record: SupportRecord = line_to_frequent_itemset(line, total_nb_of_transactions)
            support_records.append(fi_support_record)
        return support_records


def create_association_rules_from_support_records(
        transactions: List[Transaction],
        item_encoder: ItemEncoder,
        support_records: List[SupportRecord],
        # frequent_itemset_file_abs_file_name: str,
        min_confidence=0.0, min_lift=0.0) -> List[MCAR]:
    total_nb_of_transactions: int = len(transactions)

    transaction_manager = TransactionManager.create(transactions)

    mcars: List[MCAR] = []

    # with open(frequent_itemset_file_abs_file_name, "r") as frequent_itemset_file:
    #     for line in frequent_itemset_file:
    fi_support_record: SupportRecord
    for fi_support_record in support_records:
        # fi_support_record: SupportRecord = line_to_frequent_itemset(line, total_nb_of_transactions)

        ordered_statistics: OrderedStatistic
        ordered_statistics = list(
            filter_ordered_statistics(
                gen_ordered_statistics(transaction_manager, fi_support_record),
                min_confidence=min_confidence,
                min_lift=min_lift,
            )
        )
        if not ordered_statistics:
            continue
        else:
            relation_record = RelationRecord(
                fi_support_record.items, fi_support_record.support, ordered_statistics)

            support = relation_record.support
            for ordered_statistic in relation_record.ordered_statistics:
                antecedent_tmp = ordered_statistic.items_base
                consequent_tmp = ordered_statistic.items_add

                confidence = ordered_statistic.confidence
                lift = ordered_statistic.lift

                antecedent_tmp = [item_encoder.decode_item(encoding) for encoding in antecedent_tmp]
                consequent_tmp = [item_encoder.decode_item(encoding) for encoding in consequent_tmp]

                antecedent_items = [EQLiteral(*item.split(attribute_value_separator)) for item in antecedent_tmp]
                consequent_items = [EQLiteral(*item.split(attribute_value_separator)) for item in consequent_tmp]

                # antecedent_items = [Item(*item.split(attribute_value_separator)) for item in antecedent_tmp]
                # consequent_items = [Item(*item.split(attribute_value_separator)) for item in consequent_tmp]

                antecedent = GeneralizedAntecedent(antecedent_items)
                # antecedent = Antecedent(antecedent_items)
                consequent = Consequent(consequent_items)

                rule = MCAR(antecedent, consequent, support, confidence)
                mcars.append(rule)
    return mcars


def mine_MCARS_LCM(
        df: pd.DataFrame,
        encoded_transactions_abs_file_name: str,
        encoded_frequent_itemsets_abs_file_name: str,
        item_encoder_file_name: str,
        lcm_command: str,
        min_support: float = 0.1, min_confidence: float = 0.0, min_lift=0.0,
        max_length=None,
        verbose_command=False
) -> Tuple[List[MCAR], Dict[str, float]]:
    transactions_list: List[Transaction]
    item_encoder: ItemEncoder

    transactions_list, item_encoder = convert_dataset_to_transaction_file(
        dataframe=df,
        encoded_transactions_abs_file_name=encoded_transactions_abs_file_name,
        item_encoder_abs_file_name=item_encoder_file_name
    )
    total_nb_of_transactions = len(transactions_list)


    start_fim_time_s = time.time()
    lcm_command: str = lcm_frequent_itemset_mining(
        min_support=min_support,
        max_length=max_length,
        total_nb_of_transactions=total_nb_of_transactions,
        encoded_transactions_abs_file_name=encoded_transactions_abs_file_name,
        encoded_frequent_itemsets_abs_file_name=encoded_frequent_itemsets_abs_file_name,
        lcm_command=lcm_command
    )

    res = subprocess.check_output(lcm_command.split())
    if verbose_command:
        for line in res.splitlines():
            print(line)
    end_fim_time_s = time.time()

    start_assoc_time = end_fim_time_s
    support_records = read_in_frequent_itemsets(encoded_frequent_itemsets_abs_file_name)
    mcars: List[MCAR] = create_association_rules_from_support_records(
        transactions=transactions_list,
        item_encoder=item_encoder,
        support_records=support_records,
        min_confidence=min_confidence,
        min_lift=min_lift,
    )
    end_assoc_time = time.time()

    total_fim_time_s = end_fim_time_s - start_fim_time_s
    total_assoc_time_s = end_assoc_time - start_assoc_time

    timing_info = dict(
        total_fim_time_s=total_fim_time_s,
        total_assoc_time_s=total_assoc_time_s
    )

    return mcars, timing_info


if __name__ == '__main__':
    foo = line_to_frequent_itemset("2 (3)\n", 5),
    print(foo)
