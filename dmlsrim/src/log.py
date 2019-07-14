import logging
import cProfile
# import warnings

logger = logging.getLogger('DML E-Chem')
logging.basicConfig(
    level=logging.INFO,  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%d/%m/%Y %I:%M:%S %p',
    filename='./logs/dmlmung.log',

    filemode='w'  # "a"
)

# logging.debug('This message should go to the log file')
# logging.warning('Watch out!')  # will print a message to the console
# logging.info('I told you so')  # will not print anything


def start_profiling() -> 'cProfile.Profile':
    "."
    import cProfile
    prof = cProfile.Profile()
    prof.enable()
    return prof


def prof_to_stats(profile: 'cProfile.Profile') -> None:
    """."""
    import pstats
    import datetime

    now = datetime.datetime.now()
    stats = pstats.Stats(profile)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(15)
    filename = f'./logs/profile_{now.day}-{now.month}-{now.year}.prof'
    profile.dump_stats(filename)
