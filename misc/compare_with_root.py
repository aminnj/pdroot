import ROOT as r
import numpy as np
import pandas as pd
import pdroot
import time
import array
import awkward1
r.gROOT.SetBatch(1)

def make_tree(filename="jagged.root", treename="t"):
    f = r.TFile("jagged.root", "RECREATE")
    t = r.TTree(treename, "")
    branches = dict()
    def make_branch(name, tstr="vi"):
        # Python: https://docs.python.org/2/library/array.html
        # ROOT: https://root.cern.ch/doc/v612/classTTree.html
        extra = []
        if tstr == "vi": obj = r.vector("int")()
        if tstr == "vf": obj = r.vector("float")()
        if tstr == "vb": obj = r.vector("bool")()
        if tstr == "f":
            obj = array.array("f",[999]) # float
            extra.append("{}/F".format(name)) # Float_t
        if tstr == "b":
            obj = array.array("b",[0]) # signed char
            extra.append("{}/O".format(name)) # Bool_t
        if tstr == "i":
            obj = array.array("i",[999]) # signed int
            extra.append("{}/I".format(name)) # Int_t
        branches[name] = obj
        t.Branch(name,obj,*extra)


    make_branch("Jet_pt", "vf")
    make_branch("Jet_eta", "vf")
    make_branch("MET_pt", "f")

    a = awkward1.Array(
        dict(
            Jet_pt=[[42.0, 15.0, 10.5], [3.5], [11.5], [50.0, 5.0], [130., 14, 2.0, 2.0, 1.0]],
            Jet_eta=[[-2.2, 0.4, 0.5], [-5.5], [1.5], [-0.1, -3.0], [0.1, -0.1, 0.1, 5.0, 1.1]],
            MET_pt=[46.5, 30.0, 82.0, 8.9, 150.],
        )
    )
    for row in a:
        branches["Jet_pt"].clear()
        branches["Jet_eta"].clear()
        for x in row["Jet_pt"]: branches["Jet_pt"].push_back(x)
        for x in row["Jet_eta"]: branches["Jet_eta"].push_back(x)
        branches["MET_pt"][0] = row["MET_pt"]
        t.Fill()

    t.Write()
    f.Close()

def convert_root_to_python(expr):
    expr = expr.replace("&&"," and ")
    expr = expr.replace("||"," or ")
    for k in ["Sum", "Length", "Mean", "Min", "Max"]:
        expr = expr.replace(f"{k}$",f"{k.lower()}")
    return expr

def compare(ch, df, varexp_root, sel_root, varexp_python="", sel_python=""):
    print(f"Checking: {varexp_root} [{sel_root}] ... ", end="")
    if not varexp_python:
        varexp_python = convert_root_to_python(varexp_root)
    if not sel_python:
        sel_python = convert_root_to_python(sel_root)

    N = ch.Draw(varexp_root, sel_root, "goff")
    raw = ch.GetV1()
    vals_root = np.array([raw[i] for i in range(N)])
    vals_python = df.draw_to_array(varexp_python, sel_python)
    print(f"comparing {len(vals_root)} vs {len(vals_python)} elements ... ", end="")
    np.testing.assert_allclose(vals_root, vals_python)
    print("succeeded!")

if __name__ == "__main__":

    treename = "t"
    filename = "jagged.root"
    make_tree(filename)

    df = pd.read_root(filename, treename=treename)
    ch = r.TChain(treename)
    ch.Add(filename)
    print(df)

    compare(ch, df, "MET_pt + Sum$(Jet_pt*(Jet_pt>40 && abs(Jet_eta)<2.4))", "")
    compare(ch, df, "MET_pt + Sum$(Jet_pt*(Jet_pt>40 && abs(Jet_eta)<2.4))", "MET_pt>40")
    compare(ch, df, "Jet_pt", "MET_pt>40")
    compare(ch, df, "Jet_pt", "")
    compare(ch, df, "MET_pt", "")
    compare(ch, df, "Jet_pt", "Jet_pt > 40")
    compare(ch, df, "MET_pt", "MET_pt > 40")
    compare(ch, df, "Jet_pt", "MET_pt > 40")
    compare(ch, df, "Jet_pt", "abs(Jet_eta) > 1 && MET_pt > 10")
    compare(ch, df, "Jet_eta", "Jet_pt > 40 && MET_pt > 40")
    compare(ch, df, "Jet_eta + 1", "Jet_pt > 40 && MET_pt > 40")
    compare(ch, df, "Max$(abs(Jet_eta))", "")
    compare(ch, df, "Max$(abs(Jet_eta))", "MET_pt > 80")
    compare(ch, df, "Min$(Jet_pt)", "")
    compare(ch, df, "Sum$(Jet_pt)/Length$(Jet_pt)", "")
    compare(ch, df, "Length$(Jet_pt)", "")
    compare(ch, df, "Length$(Jet_pt)", "MET_pt < 10")
    compare(ch, df, "Sum$(Jet_pt)", "MET_pt < 10")
    compare(ch, df, "Jet_pt[0]", "")
    compare(ch, df, "Jet_pt[5]+Jet_eta[0]", "")
    compare(ch, df, "Sum$(Jet_pt>10)", "MET_pt>40")
    compare(ch, df, "Jet_pt", "(14. < Jet_pt) && (Jet_pt < 16.)")
    compare(ch, df, "Length$(Jet_pt) == 2 && Sum$(Jet_pt) > 40", "")
    compare(ch, df, "(MET_pt>40) && Sum$((Jet_pt>40) && (abs(Jet_eta)<2.4)) >= 1", "")
