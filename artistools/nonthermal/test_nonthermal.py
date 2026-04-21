import artistools as at

modelpath = at.get_path("testartismodel")
outputpath = at.get_path("testoutput")


def test_spencerfano() -> None:
    at.nonthermal.solvespencerfanocmd.main(
        argsraw=[], modelpath=modelpath, timedays=300, makeplot=True, npts=200, noexcitation=True, outputfile=outputpath
    )
