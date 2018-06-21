from typing import List

from subprocess import Popen, PIPE

from utils import log


def run(params: List[str]):
    """Run a command using the console / shell.
    
    :param params: List containing all parameters. The very first "parameter" is the name of the script that should be run.
    :return: 
    """
    # run
    p = Popen(params, stdout=PIPE)

    # log output
    while True:
        line = p.stdout.readline().decode('UTF-8').strip()
        if line != '':
            log.log(line)
        else:
            break
