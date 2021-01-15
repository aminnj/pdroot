import ast
import astor
from io import BytesIO
from tokenize import tokenize, NAME, ENCODING


RESERVED_TOKENS = [
    "and",
    "or",
    "abs",
    "max",
    "min",
    "sum",
    "mean",
    "not",
    "length",
    "True",
    "False",
]


def variables_in_expr(expr, exclude=RESERVED_TOKENS, include=[]):
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
        if (tokval in exclude) and (tokval not in include):
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

    # "and" -> "&"
    def visit_And(self, node):
        self.generic_visit(node)
        return ast.BitAnd()

    # "or" -> "|"
    def visit_Or(self, node):
        self.generic_visit(node)
        return ast.BitOr()

    # "not" -> "~"
    def visit_Not(self, node):
        self.generic_visit(node)
        return ast.Invert()

    # "a < b < c" -> "(a < b) and (b < c)"
    def visit_Compare(self, node):
        if len(node.ops) >= 2:
            # from pandas/core/computation/expr.py
            left = node.left
            values = []
            for op, comp in zip(node.ops, node.comparators):
                new_node = self.visit(
                    ast.Compare(comparators=[comp], left=left, ops=[op])
                )
                left = comp
                values.append(new_node)
            return self.visit(ast.BoolOp(op=ast.And(), values=values))
        self.generic_visit(node)
        return node

    # "min(x)" -> "ak.min(x, axis=-1)"
    def visit_Call(self, node):
        if hasattr(node.func, "id"):
            name = node.func.id
            if name in ["min", "max", "sum", "mean", "length"]:
                if name == "length":
                    node.func.id = "ak.count"
                else:
                    node.func.id = "ak." + node.func.id
                node.keywords.append(ast.keyword("axis", ast.Constant(-1)))
        self.generic_visit(node)
        return node

    # "x[2]" -> "ak.pad_none(x, 3)[:, 2]"
    def visit_Subscript(self, node):
        valid_slice = False
        for attr in ["value", "upper", "lower", "step"]:
            if isinstance(getattr(node.slice, attr, None), (ast.Constant, ast.Num)):
                valid_slice = True
        if valid_slice:
            if hasattr(node.slice, "value"):
                index = node.slice.value.n
                value = ast.Call(
                    func=ast.Name("ak.pad_none"),
                    args=[node.value, ast.Constant(index + 1)],
                    keywords=[],
                )
                dimslice = ast.Constant(index)
            elif hasattr(node.slice, "upper"):
                upper = node.slice.upper.n
                value = ast.Call(
                    func=ast.Name("ak.pad_none"),
                    args=[node.value, ast.Constant(upper + 1)],
                    keywords=[],
                )
                dimslice = node.slice
            node = ast.Subscript(
                value=value,
                slice=ast.ExtSlice(
                    dims=[ast.Slice(lower=None, upper=None, step=None), dimslice,]
                ),
                ctx=ast.Load(),
            )
        self.generic_visit(node)
        return node


def to_ak_expr(expr, transformer=Transformer()):
    """
    turns 
        expr = "sum(Jet_pt[abs(Jet_eta)>4.])"
    into 
        expr = "ak.sum(Jet_pt[abs(Jet_eta) > 4.0], axis=-1)"
    """
    parsed = ast.parse(expr)
    transformer.visit(parsed)
    source = astor.to_source(parsed).strip()
    return source


def split_expr_on_free_colon(expr):
    """
    When splitting on : for the purpose of drawing in 2D,
    a simple expr.split(":") won't work if it picks a slice,
    so we find a colon which has an equal number of open
    and close parentheses/brackets before it.

    Input: "sum(Jet_pt[:2]):Jet_eta"
    Output: ("sum(Jet_pt[:2])", "Jet_eta")
    """
    n_enclosure = 0
    for ic, c in enumerate(expr):
        if c == "[":
            n_enclosure += 1j
        elif c == "]":
            n_enclosure -= 1j
        elif c == "(":
            n_enclosure += 1
        elif c == ")":
            n_enclosure -= 1
        elif (c == ":") and (n_enclosure == 0):
            return expr[:ic], expr[ic + 1 :]
    return [expr]
