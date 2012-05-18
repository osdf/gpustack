"""
A dispatcher for experiments.

Given a description for a set
of experiments, disentangle every
experiment out of it and run it.
"""


import itertools
from multiprocessing import cpu_count #, Pool
from optparse import OptionParser
from time import strftime


def analyse(config):
    """
    Analyse config file _config_.
    Returns dictionary _udef_ of all
    user defined variables.
    """
    # config must end in .py (needed by __import__)
    config = config.split('.')[0]
    conf_module = __import__(config)
    udef = {}
    for key in dir(conf_module):
        if key[0] != '_':
            udef[key] = conf_module.__dict__[key]
    return udef


def component(model):
    """
    Given the description of an 'atomic'
    component, dissect the specification
    and build a schedule -- several of these
    if (i) sets are used in the component
    description or (ii) possible subcomponents
    (tuples/dicts/lists) produce several schedules.
    """
    base = {}
    tuples = []
    tuples_name = []
    for key in model.keys():
        # Found a tuple/dict/list in a 'basic' model!
        # Generate schedules for the subcomponents.
        if type(model[key]) in [tuple, dict, list]:
            schedules = disperse(model[key])
            idx = 0
            tmp = []
            # schedules is an iterable
            # any better way than this?
            for sched in schedules:
                tmp.append(sched)
            schedules = tuple(tmp)
            if schedules.__len__() > 1:
                tuples_name.append(key)
                tuples.append(schedules)
            else:
                # no use for tuple if there is only one element.
                base[key] = schedules[0]
        # sets: standard cross product
        if type(model[key]) is set:
            tuples_name.append(key)
            tuples.append(model[key])
        else:
            base[key] = model[key]
    product = itertools.product(*tuples)
    schedules = []
    for prod in product:
        tmp = base.copy()
        for idx, value in enumerate(prod):
            tmp[tuples_name[idx]] = value
        schedules.append(tmp)
    return schedules


def disperse(model):
    """
    Analyse a complex model, a sequence of
    (possibly) parrallel entities.
    """
    # Model is 'simple'
    if type(model) is dict:
        schedules = component(model)
    # Model is chain of submodels
    elif type(model) is tuple:
        chain = []
        for comp in model:
            chain.append(disperse(comp))
        schedules = itertools.product(*chain)
    # Model consists of parallel submodels
    elif type(model) is list:
        raise NotImplementedError, "\nParallel components not supported: "\
                + str(model)
    # an atomic element
    else:
        schedules = (model,)
    return schedules


def check(cores, config, model, declared):
    """
    Check for clear hints that something is wrong.
    If so, abort.
    """
    assert config.endswith(".py"), \
            "Your config file needs to be a python file, ending in .py."
    assert cores <= cpu_count(), \
            "You want to use more cores than there are in the system. \n" \
            "Don't be greedy! Sit back and rethink."
    assert model in declared.keys(), \
            "I don't know the name of your model. \n" \
            "I assumed \"" + model + "\". Could you please check?"
    assert "lab" in declared.keys(), \
            "You didn't give me a lab!\n" \
            "How should I do experiments without a lab?\n" \
            "Variable \"lab\" in your config file " \
            "must point to a .py file with lab code."


def scheduling(cores=1, config="config.py", model="model"):
    """
    Generate schedules given a configuration
    in _config_. The main model is named _model_.
    Run the set of schedules in the lab defined
    in the _config_ file.
    """
    declared = analyse(config)

    check(cores, config, model, declared)

    schedules = disperse(declared[model])

    labname = declared["lab"].split('.')[0]
    # simple logging: write some info to a file.
    # auto flush after every write (last arg = 0)
    dlog = open("log.dispatch", "w", 0)
    dlog.write("Dispatching experiments from %s.\n" % config)
    dlog.write("Cores to be used: %d.\n" % cores)
    dlog.write("Started at %s.\n" % strftime("%Y-%m-%d-%H:%M:%S"))
    dlog.write("%d Experiments.\n" % len(schedules))

    try:
        if cores == 1:
            for i, s in enumerate(schedules):
                # possible useful information 
                # for logging facilities in the lab 
                s["__lab__"] = declared["lab"]
                s["__config__"] = config

                lab = __import__(labname)
                lab.run(s)
                
                dlog.write("%d " % (i+1))
        else:
            raise NotImplementedError, "No Multicore support yet!"
    except KeyboardInterrupt:
        dlog.write("\nKeyboard Interrupt. Experiments stopped.")
    finally:
        dlog.write("\nFinished at %s.\n" % strftime("%Y-%m-%d-%H:%M:%S"))
        dlog.close()


def parse_cmd():
    """
    Parse command line. Returns parameters
    that are dubbed _meta_ here.
    """
    parser = OptionParser()
    parser.add_option("--cores", type="int", dest="cores",
            help="Number of cores that should run experiments (DEFAULT=1).",
            default=1)
    #
    parser.add_option("--cfg", dest="config",
            help="File that describes the setup of experiments \
                    (DEFAULT: config.py).",
            default="config.py")
    #
    parser.add_option("--model", dest="model",
            help="The name of the model (DEFAULT: model).",
            default="model")
    #
    (options, args) = parser.parse_args()
    # options is not a dictionary
    meta = dict()
    meta["cores"] = options.cores
    meta["config"] = options.config
    meta["model"] = options.model
    return meta


if __name__ == "__main__":
    # Handle command line arguments
    meta = parse_cmd()
    # Run experiments
    scheduling(**meta)
