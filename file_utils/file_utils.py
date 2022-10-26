import os
import sys
import pathlib
from typing import List


def read_text_file(path: str, debug=False) -> List:
    with open(path) as f:
        lines = f.read().splitlines()
    if debug:
        print("Read a file from:{}".format(path))
    return lines


def get_list_of_files(folder_path: str, file_type="*.png") -> List:
    p = pathlib.Path(folder_path)
    list_of_paths = list(p.glob(file_type))
    list_of_paths.sort()
    list_of_paths = list(map(str, list_of_paths))
    return list_of_paths
