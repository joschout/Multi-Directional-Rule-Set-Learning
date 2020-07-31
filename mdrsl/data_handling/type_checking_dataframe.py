import warnings

import pandas as pd
from pandas.core.dtypes.common import is_string_dtype


def type_check_dataframe(quant_dataframe):
    # TODO: type check
    # TODO: move to a more suitable location, e.g. code handling input data
    # if type(quant_dataframe) != QuantitativeDataFrame:
    #     raise Exception("Type of quant_dataframe must be QuantitativeDataFrame")

    # init params
    warnings.warn("Removed QuantitiveDataFrame type check, might need to be included again")

    if type(quant_dataframe) == pd.DataFrame:

        non_string_columns = []

        for column in quant_dataframe.columns:
            if not is_string_dtype(quant_dataframe[column]):
                non_string_columns.append("\t" + str(column) + ": " + str(quant_dataframe[column].dtype))
                quant_dataframe[column] = quant_dataframe[column].astype(str)

        if non_string_columns:
            non_string_columns_str = "\n".join(non_string_columns)

            warnings.warn(
                "All columns of a dataframe should be string types.\n"
                "The following columns are not stringly typed:\n" +
                non_string_columns_str +
                " Maybe you did not discretize all numerical attributes?\n"
                "CONVERTED THESE COLUMNS TO STRING!"
            )
