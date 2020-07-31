import pandas as pd

TargetAttr = str


def reorder_columns(df: pd.DataFrame, target_column: TargetAttr) -> pd.DataFrame:
    """
    Generates a dataframe with reordered columns, such that the given target column is the last column
    :param df:
    :param target_column:
    :return:
    """
    if target_column not in df.columns:
        message = f"the given target column {target_column} is not a column of the given dataframe"
        raise Exception(message)
    columns = df.columns
    reordered_columns = []
    for possibly_other_column in columns:
        if str(possibly_other_column) != str(target_column):
            reordered_columns.append(possibly_other_column)
    # reordered_columns = [other_col for other_col in columns if str(other_col) is not str(target_column)]
    reordered_columns.append(target_column)
    new_df = df[reordered_columns]
    return new_df

