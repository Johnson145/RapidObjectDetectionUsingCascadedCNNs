"""This module provides a very simple logger."""
import time
import os

log_cache = []
log_name = "log"
console_output_enabled = True


def log_set_name(name: str):
    """Set the global name of the current log.

    The name will be included in saved files and so it makes it easier to find a specific log file.

    :param name: The new name.
    :return:
    """
    global log_name
    log_name = name


def log(msg: str, log_to_console=True, log_to_file=True):
    """Add a single log line.

    :param msg: The message of the new log line.
    :param log_to_console: Whether the new log line should be shown in the console / shell.
    :param log_to_file: Whether the new log line should be appeneded to the log cache which will be used for saving log
                            files.
    :return:
    """
    global log_cache
    msg = time.strftime('%X') + ": {}".format(msg)  # prepend the current time
    if log_to_console and console_output_enabled:
        print(msg)
    if log_to_file:
        log_cache.append(time.strftime('%x') + ' ' + msg)


def log_save(directory, flush=True):
    """Save a log file with all currently-cached log lines.

    :param directory: The file path to the dir which should be used to save the new log file.
    :param flush: Whether to clear the current log cache after persisting it.
    :return:
    """
    global log_cache, log_name

    # create destination dir, if it does not exist yet
    if not os.path.exists(directory):
        os.makedirs(directory)

    import config as cf
    prefix = cf.get("session_key")

    # write to file
    f = open(directory + '/' + prefix + '-' + log_name + '.txt', 'w')
    f.write('\n'.join(log_cache))
    f.close()

    # optionally, clear the current cache
    if flush:
        log_clear()


def log_clear():
    """Clear / flush the current log cache."""
    global log_cache
    log_cache = []


def disable_console_output():
    """Do not print any further log lines to the console / shell."""
    global console_output_enabled
    console_output_enabled = False
