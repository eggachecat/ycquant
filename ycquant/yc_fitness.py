import ctypes


class YCFitness:
    @staticmethod
    def _execute(func, *args):
        return func(*args)

    def __init__(self, func_key, func_config=None, path_to_lib="libs/GPQuant.dll"):
        """

        :param func_key: str
                attr of the metric function in dll
        :param func_config: dict, default None
            config the metric function
        :param path_to_lib: str, default path_to_libs
            where is the dll
        """
        self.lib = ctypes.cdll.LoadLibrary(path_to_lib)
        self.evluate_func = getattr(self.lib, func_key)

        if func_config is None:
            func_config = {
                "argtypes": [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                             ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int],
                "restype": ctypes.c_double
            }

        for attr in func_config:
            setattr(self.evluate_func, attr, func_config[attr])

    def execute(self, func_name, *args):
        return self._execute(*args)

    def evluate(self, *args):
        return self._execute(self.evluate_func, *args)
