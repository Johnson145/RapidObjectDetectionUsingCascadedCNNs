# This file contains settings that are relevant to make this project run on a specific machine.
# Copy this file to config_local.py and adjust the following settings to your needs.
_cf = dict()

# the following dir will contain almost all additional data that is related to this project.
# this refers especially to datasets as well as to some output
_cf["project_extension_root"] = ""

# the following dir will be used to cache some relatively small files that are required very often
# if you got a SSD, you should use it here
_cf["project_extension_root_fast"] = _cf["project_extension_root"]

# you can override (almost) arbitrary options from the config.py here:
# e.g. you can set the batch size accordingly to your memory restrictions:
# _cf["batch_size"] = 150
