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
    "argmax",
    "argmin",
    "axis",
    "sum",
    "mean",
    "not",
    "length",
    "len",
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


class Transformer(ast.NodeTransformer):
    def __init__(self, aliases=dict()):
        self.aliases = aliases
        self.nreducers = 0

    def visit_Name(self, node):
        if node.id in self.aliases:
            node = self.visit(ast.parse(self.aliases[node.id]))
            return node
        self.generic_visit(node)
        return node

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
    # "min(x,y)" -> "np.minimum(x,y)"
    def visit_Call(self, node):
        if hasattr(node.func, "id"):
            name = node.func.id
            if name in [
                "min",
                "max",
                "sum",
                "mean",
                "length",
                "len",
                "argmin",
                "argmax",
            ]:
                if len(node.args) == 1:
                    if name in ["length", "len"]:
                        node.func.id = "count"
                    node.func.id = "ak." + node.func.id
                    node.keywords.append(ast.keyword("axis", ast.Constant(-1)))
                    if name in ["argmin", "argmax"]:
                        node.keywords.append(ast.keyword("keepdims", ast.Constant(True)))
                    self.nreducers += 1
                elif (len(node.args) == 2) and name in ["min", "max"]:
                    node.func.id = {"min": "np.minimum", "max": "np.maximum"}[name]
                else:
                    raise Exception(
                        f"Unsupported function '{name}' with {len(node.args)} arguments."
                    )
        self.generic_visit(node)
        return node

    # "x[2]" -> "ak.pad_none(x, 3, clip=True)[:, 2]"
    def visit_Subscript(self, node):
        valid_slice = False
        s = node.slice

        # if the slice is a negative number, then convert it into a simpler form
        # so that it can be subsequently parsed the same way as positive numbers
        if hasattr(s, "value") and isinstance(s.value, ast.UnaryOp):
            if isinstance(s.value.op, ast.USub) and isinstance(
                s.value.operand, ast.Num
            ):
                n = -s.value.operand.n
                if n != -1:
                    raise Exception(
                        f"Negative index `{n}` not supported. "
                        "Only -1 supported, due to ak.pad_none() padding the end of an array."
                    )
                s = ast.Index(value=ast.Num(n))

        for attr in ["value", "upper", "lower", "step"]:
            if isinstance(getattr(s, attr, None), (ast.Constant, ast.Num)):
                valid_slice = True
        if valid_slice:
            if hasattr(s, "value"):
                upper = s.value.n
                dimslice = ast.Constant(upper)
                self.nreducers += 1
            elif hasattr(s, "upper"):
                upper = s.upper.n
                dimslice = s
            else:
                raise Exception(f"Slice node not supported: {s}")
            absupper = abs(upper)
            if upper >= 0:
                clip = True
                absupper = ast.Constant(absupper + 1)
            else:
                clip = False
                absupper = ast.Constant(absupper)
            value = ast.Call(
                func=ast.Name("ak.pad_none"),
                args=[node.value, absupper],
                keywords=[ast.keyword("clip", ast.Constant(clip))],
            )
            node = ast.Subscript(
                value=value,
                slice=ast.ExtSlice(
                    dims=[ast.Slice(lower=None, upper=None, step=None), dimslice,]
                ),
                ctx=ast.Load(),
            )
        # else:
        #     # for when an index array is used as a slice
        #     # currently will only work if that array has a None
        #     # (https://github.com/scikit-hep/awkward-1.0/issues/708)
        #     new_slice = ast.Call(
        #         func=ast.Name("ak.singletons"), args=[node.slice], keywords=[],
        #     )
        #     node = ast.Subscript(value=node.value, slice=new_slice, ctx=ast.Load())
        self.generic_visit(node)
        return node


def to_ak_expr(expr, aliases=dict(), transformer=Transformer()):
    """
    turns 
        expr = "sum(Jet_pt[abs(Jet_eta)>4.])"
    into 
        expr = "ak.sum(Jet_pt[abs(Jet_eta) > 4.0], axis=-1)"
    """
    transformer.aliases = aliases
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
