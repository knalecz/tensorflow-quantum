import sympy


def generate_sympy_parameters(p):
    return sympy.symbols("parameter_:%d" % (2 * p))
