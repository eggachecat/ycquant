import re


class YCInterpreter:
    __MC_map = {
        "add": "CELAdd",
        "div": "CELDivide",
        "mul": "CELMultiply",
        "sub": "CELSubtract",
        "sin": "CELSine"
    }

    @staticmethod
    def __MC_convert_variable_format(match):
        _var = match[0]
        return "v1[{n}]".format(n=1 + int(_var[1:]))

    def __init__(self):
        pass

    @staticmethod
    def mc_interprete(command, _dict=None, convert_func=None):

        if _dict is None:
            _dict = YCInterpreter.__MC_map

        if convert_func is None:
            convert_func = YCInterpreter.__MC_convert_variable_format

        new_command = command
        for key in _dict:
            new_command = new_command.replace(key, _dict[key])

        return re.sub("X[0-9]+", repl=convert_func, string=new_command)
