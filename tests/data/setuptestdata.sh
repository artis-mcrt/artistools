#!/usr/bin/env zsh

# the user runs this script with "source", thus a plain "set -e" would leave the option on in the
# shell of the user. A subshell function keeps the option inside the function. The status of the
# function reaches the caller, and an error stops no interactive shell
setuptestdata() (
    # set -e stops the script at the first error. A download that fails would otherwise leave an
    # empty model folder. Each test that reads that folder then reports a missing file, and not
    # the true cause
    set -ex

    RELEASEURL=https://github.com/artis-mcrt/artistools/releases/download/v2026.5.9

    get_archive() {
        # curl can leave a partial archive after an error. Without the rm call, the test for the file
        # then skips the download
        if [ ! -f "$1" ]; then curl --fail --retry 3 -O -L "$RELEASEURL/$1" || rm -f "$1"; fi
        # a partial archive or an absent one stops the script here, before the rm call below
        tar -tf "$1" > /dev/null
    }

    get_archive testmodel.tar.xz
    rm -rf testmodel/
    mkdir -p testmodel/
    tar -xf testmodel.tar.xz --directory testmodel/
    # find testmodel -size +1M -exec xz -v {} \;

    get_archive vspecpolmodel.tar.xz
    tar -xf vspecpolmodel.tar.xz

    get_archive test_classicmode_3d.tar.xz
    tar -xf test_classicmode_3d.tar.xz

    # git holds this archive, because it is small. Thus it needs no download
    tar -xf test_classicmode_1d.tar.xz

    cp grid.out testmodel/

    xz -d testmodel/transitiondata.txt.xz
)

setuptestdata
