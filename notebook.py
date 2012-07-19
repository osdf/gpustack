"""
Take care of managing experiments: storing, retrieving.
"""


import sys
sys.path.append("../")


import types
import hashlib, shutil
import cPickle
import os, glob
from os.path import join, exists
from time import strftime
import json


def func_str(func):
    """
    convert a function into a string.
    Standard function string comes with
    address of function, avoid that,
    for hashing reasons. Also look at
    bytecode of function. Possibly not
    optimal solution.
    """
    fct_hash = str(hash(str(func.__code__.co_code)))
    return "<" + str(func).split()[1] + "-" + fct_hash + ">"


def gen_str(gen):
    """
    """
    g_hash = str(hash(str(gen.gi_code.co_code)))
    hsh = "<" + str(gen).split()[2] + "-" + g_hash + ">"
    return hsh


def tuple_str(tup):
    """
    Convert a tuple to string.
    """
    stringed = "("
    for elems in tup:
        stringed += atom_str(elems) + ","
    stringed += ")"
    return stringed


def dict_str(dic):
    """
    Convert a dictionary to a hashable
    string. Avoid arbitrarines by sorting
    keys.
    """
    stringed = "{"
    keys = dic.keys()
    keys.sort()
    for k in keys:
        stringed += str(k) + ": " + atom_str(dic[k]) + "\n"
    stringed += "}"
    return stringed


def atom_str(atom):
    """
    Given an element, make a nice string out of it.
    """
    if type(atom) is types.FunctionType:
        stringed = func_str(atom)
    elif type(atom) is types.GeneratorType:
        stringed = gen_str(atom)
    elif type(atom) is dict:
        stringed = dict_str(atom)
    elif type(atom) is tuple:
        stringed = tuple_str(atom)
    else:
        stringed = str(atom)
    return stringed


def prepare(schedule, depot="depot") :
    """
    Prepare things for notebook.

    - generate a directory out of a schedule-hash
    - copy over labfile and config file
    - dump (cPickle) schedule
    """
    # _now_ is the time right now
    now = strftime("%Y-%m-%d-%H:%M:%S")
    # hash the lab file: if this file is changed,
    # consider it a different experiment
    # "__lab__" key was put by dispatch.py
    lab = open(schedule["__lab__"])
    schedule["__lab__#"] = hashlib.sha1(lab.read()).hexdigest()
    # Note: Don't hash the config file. A config file
    # can define several single experiments. A specific 
    # experiment may come from two different config files.
    # Instead, hash the schedule.
    to_hash = dict_str(schedule)
    folder = hashlib.sha1(to_hash).hexdigest()[0:20]
    path = join(depot, folder)
    if not exists(path):
        os.makedirs(path)
        schedfile = join(path, now) + ".schedule"
        with open(schedfile, "w") as f:
            f.write(json.dumps(to_hash))
    else:
        print "[NOTEBOOK:prepare] It seems, you are _re_running an experiment."
        print "[NOTEBOOK:prepare] Found already this hash:", folder
        print "[NOTEBOOK:prepare] _schedule_ will not be jsonified.\n"
    # get config and lab files
    path = join(path, now) 
    shutil.copy(schedule["__lab__"], path + ".lab")
    # even though we didn't hash the config file, it is good
    # to know, which config's gave rise to this experiment.
    shutil.copy(schedule["__config__"], path + ".config")
    print "[NOTEBOOK:prepare] Working on %s\n"%path
    return path
