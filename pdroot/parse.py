import ast
import astor
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
        if tokval in ["and", "or", "abs", "max", "min", "sum", "mean"]:
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


def sandwich_vars_in_expr(expr, prefix="", suffix=""):
    """
    prepends `prefix`, and appends `suffix`
    to variables in an expression string
    """

    varnames = []
    g = list(tokenize(BytesIO(expr.encode("utf-8")).readline))
    buff = ""
    varnames = []
    for ix, x in enumerate(g):
        toknum, tokval = x[:2]
        if toknum in [ENCODING]:
            continue
        if toknum != NAME:
            buff += tokval
            continue
        if tokval in ["and", "or", "abs", "max", "min", "sum"]:
            buff += f" {tokval} "
            continue
        else:
            buff += f"{prefix}{tokval}{suffix}"
        varnames.append(tokval)
    return buff, varnames


class Transformer(ast.NodeTransformer):
    def visit_And(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        return ast.BitAnd()

    def visit_Or(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        return ast.BitOr()

    def visit_Call(self, node):
        if hasattr(node.func, "id"):
            name = node.func.id
            if name in ["min", "max", "sum", "mean", "length"]:
                if name == "length":
                    node.func.id = "ak.count"
                else:
                    node.func.id = "ak." + node.func.id
                node.keywords.append(ast.keyword("axis", ast.Num(-1)))
        ast.NodeVisitor.generic_visit(self, node)
        return node


def to_ak_expr(expr):
    """
    turns 
        expr = "sum(Jet_pt[abs(Jet_eta)>4.])"
    into 
        expr = "ak.sum(Jet_pt[abs(Jet_eta) > 4.0], axis=-1)"
    """
    parsed = ast.parse(expr)
    Transformer().visit(parsed)
    source = astor.to_source(parsed).strip()
    return source
