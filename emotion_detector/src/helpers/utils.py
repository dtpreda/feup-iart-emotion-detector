import sys
import os


def block_print():
    '''Overwrite standart output to block prints'''
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    '''Reenable prints'''
    sys.stdout = sys.__stdout__


def remove_empty_lines(str):
    '''Remove empty lines from string'''
    lines = str.split("\n")
    non_empty_lines = [line for line in lines if line.strip() != ""]

    res = ""
    for line in non_empty_lines:
        res += line + "\n"

    return res
