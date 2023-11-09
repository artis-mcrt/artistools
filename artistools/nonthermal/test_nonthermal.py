import artistools as at

modelpath = at.get_config()["path_testartismodel"]
outputpath = at.get_config()["path_testoutput"]


def test_nonthermal() -> None:
    at.nonthermal.plot(argsraw=[], modelpath=modelpath, outputfile=outputpath, timestep=70)


def test_spencerfano() -> None:
    at.nonthermal.solvespencerfanocmd.main(
        argsraw=[], modelpath=modelpath, timedays=300, makeplot=True, npts=200, noexcitation=True, outputfile=outputpath
    )
