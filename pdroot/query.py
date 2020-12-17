def hacky_query_eval(df, varstr, selstr="", verbose=False):
    """
    Please don't read/use. This is dangerous and stupid, kind of like
    integrating a function by printing out a plot, coloring the area under it in red,
    faxing it to yourself, then counting red pixels to get the area.

    Basically I wanted some way to convert

        df.query("dimuon_mass > 5 and pass_baseline_iso").eval("dimuon_mass").mean()

    into

        df["dimuon_mass"][ (df["dimuon_mass"] > 5) & (df["pass_baseline_iso"]) ].mean()

    because the latter doesn't make an intermediate copy of all the columns with query(),
    and it also doesn't do jitting with numexpr. In principle, this is much faster to execute.

    Usage:

        arr = hacky_query_eval(
            df_data,
            varstr = "dimuon_mass",
            selstr = "pass_baseline_iso and 0<logabsetaphi<1.25",
        )
        print(arr.mean())
    """
    from pandas.core.computation.expr import Expr
    from pandas.core.computation.scope import Scope

    env = Scope(
        1, global_dict=globals(), local_dict=locals(), resolvers=[df], target=None,
    )

    def inject_df(s):
        """
        convert expression string like (a > 1) to (df["a"] > 1)
        so that it can be eval'd later
        """
        expr = Expr(s, env=env, parser="pandas")
        self = expr._visitor

        def visit_Name_hack(node, **kwargs):
            result = self.term_type(node.id, self.env, **kwargs)
            result._name = f'df["{result._name}"]'
            return result

        def _maybe_downcast_constants_hack(left, right):
            return left, right

        expr._visitor.visit_Name = visit_Name_hack
        expr._visitor._maybe_downcast_constants = _maybe_downcast_constants_hack
        expr.terms = expr.parse()
        return str(expr)

    varexpr = inject_df(varstr)
    toeval = f"({varexpr})"
    if selstr:
        selexpr = str(inject_df(selstr))
        toeval += f"[{selexpr}].values"
    if verbose:
        print(f"Evaluating string: {toeval}")
    result = eval(toeval)
    return result
