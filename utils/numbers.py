def is_number(arbitrary_var):
    """Check whether arbitrary_var represents a number.

    :param arbitrary_var:
    :return:
    """
    try:
        float(arbitrary_var)
        return True
    except TypeError:
        return False
    except ValueError:
        return False
