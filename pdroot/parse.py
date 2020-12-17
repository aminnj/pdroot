from io import BytesIO
from tokenize import tokenize, NAME, ENCODING

def variables_in_expr(expr):
    """
    Given a string like "DV_x:DV_y:(lxy < DV_x+1) and (lxy>1)", returns a list of
    ["DV_x", "DV_y", "lxy"]
    (i.e., extracts what seem to be column names)
    """

    varnames = []
    g = list(tokenize(BytesIO(expr.encode("utf-8")).readline))
    for ix, x in enumerate(g):
        toknum = x[0]
        tokval = x[1]
        if toknum != NAME:
            continue
        if ix > 0 and g[ix - 1][1] in ["."]:
            continue
        if ix < len(g) - 1 and g[ix + 1][1] in [".", "("]:
            continue
        if tokval in ["and", "or"]:
            continue
        varnames.append(tokval)
    varnames = list(set(varnames))
    return varnames

def nops_in_expr(expr):
    """
    Number of mathematical/logical operations in an expression
    """

    varnames = []
    from tokenize import tokenize, NAME, OP
    from io import BytesIO

    tokens = tokenize(BytesIO(expr.encode("utf-8")).readline)
    nops = 0
    for x in tokens:
        toknum, tokval = x[:2]
        nops += (toknum == NAME) and (tokval in ["and", "or"])
        nops += toknum == OP
    return nops

def suffix_vars_in_expr(expr, suffix):
    """
    appends `suffix` to variables in an expression string
    """

    varnames = []
    g = list(tokenize(BytesIO(expr.encode("utf-8")).readline))
    buff = ""
    varnames = []
    for ix, x in enumerate(g):
        toknum, tokval = x[:2]
        if toknum not in [ENCODING]:
            buff += tokval
        if toknum == NAME:
            buff += suffix
        if toknum != NAME:
            continue
        if ix > 0 and g[ix - 1][1] in ["."]:
            continue
        if ix < len(g) - 1 and g[ix + 1][1] in [".", "("]:
            continue
        varnames.append(tokval)
    return buff, varnames

