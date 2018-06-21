"""
This module provides information about the individual object classes which have actually been used during runtime.

It has originally been designed for a multi-class problem. Currently, only binary classification is used though.
"""
################## Private Module State #######################
_all_labels_by_key = {}
_all_labels_by_iid = {}
_last_used_iid = 1  # must equal the maximum predefined iid

# for a binary use case, the first two ids have fixed meanings
IID_BACKGROUND = 0
IID_FOREGROUND = 1
KEY_BACKGROUND = "background"
KEY_FOREGROUND = "foreground"


class Label:
    """An arbitrary label describing one type of classifiable objects."""

    def __init__(self, iid: int, key: str):
        self._iid = iid
        self._key = key

    @property
    def iid(self) -> int:
        return self._iid

    @property
    def name(self) -> str:
        return self.key.title()

    @property
    def key(self) -> str:
        """Get a readable, but unique key for this label."""
        return self._key

    @property
    def is_background(self) -> bool:
        return self.iid == IID_BACKGROUND

    @property
    def is_foreground(self) -> bool:
        return self.iid == IID_FOREGROUND

################## Public Module API ##########################

def get_by_key(label_key: str) -> Label:
    """Get a label object by its key.
    For the same key, the same object will be returned.
    Will create the label, if it does not exist yet.
    """
    # if the label_key wasn't used yet, we need to create a new object
    if label_key not in _all_labels_by_key:
        # first, we need the iid
        # check for special cases or assign a new one
        if label_key == KEY_BACKGROUND:
            label_iid = IID_BACKGROUND
        elif label_key == KEY_FOREGROUND:
            label_iid = IID_FOREGROUND
        else:
            global _last_used_iid
            _last_used_iid += 1
            label_iid = _last_used_iid

        # create and save the new label
        label = Label(label_iid, label_key)
        _all_labels_by_key[label_key] = label
        _all_labels_by_iid[label_iid] = label

    return _all_labels_by_key[label_key]


def get_by_iid(label_iid: int) -> Label:
    """Get a label object by its iid.
    For the same iid, the same object will be returned.
    The label must exist already or have a pre-known iid, otherwise an error will be raised.
    """
    if label_iid not in _all_labels_by_iid:
        if label_iid == IID_BACKGROUND:
            label_key = KEY_BACKGROUND
        elif label_iid == IID_FOREGROUND:
            label_key = KEY_FOREGROUND
        else:
            raise ValueError("Found unknown label_iid: {}".format(label_iid))

        # create and save the new label
        label = Label(label_iid, label_key)
        _all_labels_by_key[label_key] = label
        _all_labels_by_iid[label_iid] = label

    return _all_labels_by_iid[label_iid]


def n_labels() -> int:
    """Get the total number of used labels."""
    return len(_all_labels_by_key)
