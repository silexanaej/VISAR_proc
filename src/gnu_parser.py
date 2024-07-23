"""
Temporary file to test new parsing function
written in Python in order to avoid bash dependency.
"""

import io
import mmap
import regex as re

def grep(pattern, file_path):  
    end_str = '.*\n'
    pattern = pattern + end_str
    pattern = pattern.encode("ascii")
    with io.open(file_path, "r", encoding="ascii") as f:
        match = re.search(pattern, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))[0].decode('ascii')  # grep one line after match
    return match.split('\n')[0]  # keeping first line

def grep_first(pattern, file_path):
    pattern = pattern.encode("ascii")
    with io.open(file_path, "r", encoding="ascii") as f:
        return re.search(pattern, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))[0].decode('ascii')  # grep first occurence

def grep_all(pattern, file_path):
    pattern = pattern.encode("ascii")
    with io.open(file_path, "r", encoding="ascii") as f:
        list_occ_bin = re.findall(pattern, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))  # grep all occurences but returns a list of binary strings
    list_occ_str = [ii.decode("ascii") for ii in list_occ_bin]
    return list_occ_str

def grep_count(pattern, file_path):
    pattern = pattern.encode("ascii")
    with io.open(file_path, "r", encoding="ascii") as f:
        return len(re.findall(pattern, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)))  # grep number of occurences

def grep_nlines_after(pattern, file_path, nb_lines):  
    # equivalent to a grep -An -m1 'expr' filename | sed '/expr/d' 
    end_str = '.*\n'
    for ii in range(0, nb_lines):
        end_str += '.*\n'
    pattern = pattern + end_str
    pattern = pattern.encode("ascii")
    with io.open(file_path, "r", encoding="ascii") as f:
        match = re.search(pattern, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))[0].decode('ascii')  # grep nth lines after match
    return match.split('\n')[1:-1]  # removing pattern line and last '' from the match. Each element is a str corresponding to the line.

def get_nth_line(file_path, nth):
    if nth > 20000: # readlines reads entire file but is faster if line number is large.
        line = open(file_path, "r").readlines()[nth-1]
        return [str(line[:-1])]
    else:
        end_str = '.*\n'
        for ii in range(0, nth-1):
            end_str += '.*\n'
        pattern = end_str
        pattern = pattern.encode("ascii")
        with io.open(file_path, "r", encoding="ascii") as f:
            match = re.search(pattern, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))[0].decode('ascii')
        return [str(match.split('\n')[:-1][-1])]  # removing last '' from the match. Each element is a str corresponding to the line.
