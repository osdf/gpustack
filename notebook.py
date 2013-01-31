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
import fnmatch


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


def pick_enable(elm):
    """
    """
    if type(elm) is dict:
        copy = {}
        keys = elm.keys()
        for k in keys:
            copy[k] = pick_enable(elm[k])
        return copy
    
    if type(elm) is tuple:
        copy = []
        for e in elm:
            copy.append(pick_enable(e))
        return tuple(copy)

    if type(elm) is types.GeneratorType:
        return gen_str(elm)

    return elm


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
    pick = pick_enable(schedule)
    folder = hashlib.sha1(to_hash).hexdigest()[0:20]
    path = join(depot, folder)
    if not exists(path):
        os.makedirs(path)
        schedfile = join(path, now) + ".schedule"
        with open(schedfile, "w") as f:
            cPickle.dump(pick, f)
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
    print "[NOTEBOOK:prepare] Working on %s\n" % path
    return path


def reload(depot, folder, tag, layer):
    """
    """
    import utils
    
    cwd = os.getcwd()
    path = join(depot, folder)
    os.chdir(path)

    for f in os.listdir('.'):
        if f.endswith("schedule"):
            sched_f = f
    sched_f = open(sched_f)
    sched = cPickle.load(sched_f)
    sched_f.close()

    if layer >= 0:
        ltype = sched['stack'][layer]['type']
        params = utils.load_params(tag + ".params")
        shape = params[layer]['shape']
        model = ltype.__new__(ltype)
        model.__init__(shape=shape, **sched)
        model.reload(params[layer]['params'])
    os.chdir(cwd)

    return model, sched


def clean_up(depot, ending=".roc.pickle", files=None):
    """
    Clean up _depot_: All folders that are
    having no file ending in _ending_ are
    deleted.
    """
    if files is None:
        files = []
        for root, dn, fnames in os.walk(depot):
            if root == depot:
                continue
            if len(fnmatch.filter(fnames, "*"+ending)) == 0:
                files.append(root)
        return files
    else:
        for f in files:
            shutil.rmtree(f)


def roc_fp95(depot, folders=None):
    """
    Show false positive rate of ROC at 95%
    true positives.
    """
    if folders is None:
        folders = []
        for f in os.listdir(depot):
            folders.append(f)
    res = {}
    for f in folders:
        path = join(depot, f)
        for match in glob.glob(path+"/*.roc.pickle"):
            rocf = open(match)
            roc = cPickle.load(rocf)
            for e in roc.keys():
                eset = roc[e]
                for size in eset.keys():
                    sset = eset[size]
                    for dist in sset.keys():
                        fp95 = sset[dist]['fp_at_95']
                        if e in res:
                            best = res[e].keys()[0]
                            if fp95 < best:
                                res[e] = {fp95: [f, size, dist]}
                        else:
                            res[e] = {fp95: [f, size, dist]}
    return res


def grep_log(logfile, field, constraints={}):
    f = open(logfile)
    grep = {}
    res = []
    for line in f:
        jline = json.loads(line)
        valid = True
        for c in constraints:
            if jline[c] != constraints[c]:
                valid = False
                break
        if valid:
            res.append(jline[field])
    return res
