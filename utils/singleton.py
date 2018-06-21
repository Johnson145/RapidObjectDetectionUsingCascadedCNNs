class Singleton(type):
    """This class can be used as a meta class for any other class that should implement the Singleton pattern.

    You can use it e.g. for the class "Foo" as follows:
    class Foo(metaclass=abc.ABCMeta):
        [..]

    Afterwards, calling the Foo constructor will create a single Foo instance on the very first call. Each proceeding
    call will provide you the same instance again.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
