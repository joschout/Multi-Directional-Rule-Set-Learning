from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


class LiteralCache:
    """class which stores __literals
    and corresponding truth values
    e.g. [
        "food=banana": [True, True, False, False, True],
        "food=apple" : [True, True, True, True, False]
    ]

    """

    def __init__(self):
        self.__cache = {}  # type: Dict[str, np.ndarray]

    def insert(self, literal, truth_values):
        self.__cache[literal] = truth_values

    def get(self, literal: str) -> np.ndarray:
        return self.__cache[literal]

    def __contains__(self, literal: str):
        """function for using in
        on LiteralCache object
        """

        return literal in self.__cache.keys()


class QuantitativeDataFrame:

    def __init__(self, dataframe):
        if type(dataframe) != pd.DataFrame:
            raise Exception("type of dataframe must be pandas.dataframe")

        self.__dataframe = dataframe
        # make the class column stringly typed
        self.__dataframe.iloc[:, -1] = self.__dataframe.iloc[:, -1].astype(str)

        # Dict mapping
        #    each column name to
        #    the a numpy array of the sorted and unique values of that column
        self.__preprocessed_columns = self.__preprocess_columns(dataframe)  # type: Dict[str, np.ndarray]

        # literal cache for computing rule statistics
        # - support and confidence
        self.__literal_cache = LiteralCache()

        # so that it doesn't have to be computed over and over
        self.size = dataframe.index.size  # type: int

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.__dataframe

    def column(self, colname: str) -> np.ndarray:
        return self.__preprocessed_columns[colname]

    def mask(self, vals):
        return self.__dataframe[vals]

    def find_covered_by_antecedent_mask(self, antecedent) -> np.ndarray:
        """
        returns:
            mask - an array of boolean values indicating which instances
            are covered by antecedent
        """

        # todo: compute only once to make function faster
        dataset_size = self.size  # type: int

        cummulated_mask = np.ones(dataset_size).astype(bool)  # type: np.ndarray

        for literal in antecedent:
            current_mask = self.find_covered_by_literal_mask(literal)  # type: np.ndarray
            cummulated_mask &= current_mask

        return cummulated_mask

    def find_covered_by_literal_mask(self, literal):
        """
        returns:
            mask - an array of boolean values indicating which instances
            are covered by literal
        """
        dataset_size = self.size

        attribute, interval = literal

        # the column that concerns the
        # iterated attribute
        # instead of pandas.Series, grab the ndarray
        # using values attribute
        relevant_column = self.__dataframe[[attribute]].values.reshape(dataset_size)  # type: np.ndarray

        # this tells us which instances satisfy the literal
        current_mask = self.get_literal_coverage(literal, relevant_column)  # type: np.ndarray
        return current_mask

    def find_covered_by_rule_mask(self, rule) -> Tuple[np.ndarray, np.ndarray]:
        """
        returns:
            covered_by_antecedent_mask:
                - array of boolean values indicating which
                dataset rows satisfy antecedent

            covered_by_consequent_mask:
                - array of boolean values indicating which
                dataset rows satisfy conseqeunt
        """

        dataset_size = self.__dataframe.index.size  # type: int

        # initialize a mask filled with True values
        # it will get modified as futher __literals get
        # tested

        # for optimization - create cummulated mask once
        # in constructor

        instances_satisfying_antecedent_mask = self.find_covered_by_antecedent_mask(rule.antecedent)

        instances_satisfying_consequent_mask = self.__get_consequent_coverage_mask(rule)
        instances_satisfying_consequent_mask = instances_satisfying_consequent_mask.reshape(dataset_size)

        return instances_satisfying_antecedent_mask, instances_satisfying_consequent_mask

    def calculate_rule_statistics(self, rule):
        """calculates rule's confidence and
        support using efficient numpy functions


        returns:
        --------

            support:
                float

            confidence:
                float
        """

        dataset_size = self.__dataframe.index.size

        # initialize a mask filled with True values
        # it will get modified as futher __literals get
        # tested

        # for optimization - create cummulated mask once
        # in constructor
        cummulated_mask = np.array([True] * dataset_size)

        for literal in rule.antecedent:
            attribute, interval = literal

            # the column that concerns the
            # iterated attribute
            # instead of pandas.Series, grab the ndarray
            # using values attribute
            relevant_column = self.__dataframe[[attribute]].values.reshape(dataset_size)

            # this tells us which instances satisfy the literal
            current_mask = self.get_literal_coverage(literal, relevant_column)

            # add cummulated and current mask using logical AND
            cummulated_mask &= current_mask

        instances_satisfying_antecedent = self.__dataframe[cummulated_mask].index
        instances_satisfying_antecedent_count = instances_satisfying_antecedent.size

        # using cummulated mask to filter out instances that satisfy consequent
        # but do not satisfy antecedent
        instances_satisfying_consequent_mask = self.__get_consequent_coverage_mask(rule)
        instances_satisfying_consequent_mask = instances_satisfying_consequent_mask.reshape(dataset_size)

        instances_satisfying_consequent_and_antecedent = self.__dataframe[
            instances_satisfying_consequent_mask & cummulated_mask
            ].index

        instances_satisfying_consequent_and_antecedent_count = instances_satisfying_consequent_and_antecedent.size
        instances_satisfying_consequent_count = self.__dataframe[instances_satisfying_consequent_mask].index.size

        # instances satisfying consequent both antecedent and consequent

        support = instances_satisfying_antecedent_count / dataset_size

        confidence = 0
        if instances_satisfying_antecedent_count != 0:
            confidence = instances_satisfying_consequent_and_antecedent_count / instances_satisfying_antecedent_count

        return support, confidence

    def __get_consequent_coverage_mask(self, rule):
        consequent = rule.consequent
        attribute, value = consequent

        class_column = self.__dataframe[[attribute]].values  # type: np.ndarray
        class_column = class_column.astype(str)

        literal_key = "{}={}".format(attribute, value)

        mask = []

        if literal_key in self.__literal_cache:
            mask = self.__literal_cache.get(literal_key)
        else:
            mask = class_column == value

        return mask

    def get_literal_coverage(self, literal: Tuple[str, object], values: np.ndarray) -> np.ndarray:
        """
        returns mask which describes the instances that
        satisfy the interval (literal)

        function uses cached results for efficiency

        literal: the literal to check with
        values: the column values as an np.ndarray

        """

        if type(values) != np.ndarray:
            raise Exception("Type of values must be numpy.ndarray")

        attribute, interval = literal

        literal_key = "{}={}".format(attribute, interval)

        # check if the result is already cached, otherwise
        # calculate and save the result
        if literal_key in self.__literal_cache:
            mask = self.__literal_cache.get(literal_key)  # type: np.ndarray
        else:

            if type(interval) == str:
                mask = np.array([val == interval for val in values])  # type: np.ndarray
            else:
                mask = interval.test_membership(values)  # type: np.ndarray

            self.__literal_cache.insert(literal_key, mask)

        # reshape mask into single dimension
        mask = mask.reshape(values.size)  # type: np.ndarray

        return mask

    def __preprocess_columns(self, dataframe):

        # Covert to dictionary  {column: list of column values}
        # need to convert it to numpy array
        dataframe_dict = dataframe.to_dict(orient="list")  # type: Dict[str, List[object]]

        dict_of_column_to_nd_array_of_possible_values = {}  # type: Dict[str, np.ndarray]

        for column, value_list in dataframe_dict.items():
            unique_sorted_column_values = np.unique(value_list)
            dict_of_column_to_nd_array_of_possible_values[column] = unique_sorted_column_values

        return dict_of_column_to_nd_array_of_possible_values


if __name__ == '__main__':
    df = pd.DataFrame({'col1': [1, 2],
                       'col2': [0.5, 0.75]},
                      index=['row1', 'row2'])

    print(df)
    print(df.to_dict(orient="list"))

    foobar = df[['col1']].values
    mask = foobar == 1

    qdf = QuantitativeDataFrame(df)
    print(qdf)