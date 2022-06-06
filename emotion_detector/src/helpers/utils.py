import sys
import os

# Disable


def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore


def enablePrint():
    sys.stdout = sys.__stdout__


def removeEmptyLines(str):
    lines = str.split("\n")
    non_empty_lines = [line for line in lines if line.strip() != ""]

    res = ""
    for line in non_empty_lines:
        res += line + "\n"

    return res
