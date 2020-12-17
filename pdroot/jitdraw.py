import functools
import numba
import numpy as np

from yahist import Hist1D

from .parse import suffix_vars_in_expr

numba.set_num_threads(min(numba.get_num_threads(), 4))

@numba.njit()
def compute_bin_1d_uniform(x, nbins, b_min, b_max, overflow=False):
    if overflow:
        if x > b_max:
            return nbins - 1
        elif x < b_min:
            return 0
    ibin = int(nbins * (x - b_min) / (b_max - b_min))
    if x < b_min or x > b_max:
        return -1
    else:
        return ibin

def get_executable_str_and_vars(varexp, cut):
    cut_suffix, cut_vars = suffix_vars_in_expr(cut, suffix="[i]")
    varexp_suffix, varexp_vars = suffix_vars_in_expr(varexp, suffix="[i]")

    vars_all = sorted(list(set(cut_vars + varexp_vars)))
    vars_comma_sep = ",".join(vars_all)
    vars_first = vars_all[0]

    template = f"""
@numba.jit(nopython=True, fastmath=True, nogil=True, parallel=True)
def temp_func({vars_comma_sep}, bins):
    b_min = bins[0]
    b_max = bins[-1]
    nbins = bins.shape[0] - 1
    hist = np.zeros(nbins, dtype=np.float64)
    for i in numba.prange(len({vars_first})):
        if {cut_suffix}:
            value = {varexp_suffix}
            ibin = compute_bin_1d_uniform(value, nbins, b_min, b_max)
            if ibin >= 0:
                hist[ibin] += 1
    return hist
    """
    return template, vars_all

def string_to_function(s):
    """
    takes string containing function def via `def`
    and returns function object
    need numba, numpy to be in global namespace for jitting
    """
    defline = filter(lambda x: x.startswith("def "), s.strip().splitlines()[:2])
    defline = list(defline)[0]
    funcname = defline.split(" ",1)[1].split("(",1)[0]
    exec(s)
    return locals()[funcname]


@functools.lru_cache(maxsize=128)
def get_jitfunc_and_vars(varexp, cut):
    s, vars_all = get_executable_str_and_vars(varexp, cut)
    func = string_to_function(s)
    return func, vars_all

def jitdraw(df, varexp, cut, bins=np.linspace(0, 1, 10)):
    f, vars_all = get_jitfunc_and_vars(varexp, cut)
    to_eval = "f(" + ", ".join([f'df["{v}"].values' for v in vars_all]) + ", bins" + ")"
    return Hist1D.from_bincounts(eval(to_eval), bins)
