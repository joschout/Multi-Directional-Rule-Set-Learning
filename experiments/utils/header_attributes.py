from typing import List

TargetAttr = str


def get_header_attributes(abs_file_name: str) -> List[str]:
    header_attributes: List[str]
    with open(abs_file_name, 'r') as input_data_file:
        header_line: str = input_data_file.readline()
        header_line_without_line_break: str = header_line.replace('\n', '')
        header_attributes = header_line_without_line_break.split(',')
    return header_attributes

