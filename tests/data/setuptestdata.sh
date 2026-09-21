#!/usr/bin/env zsh

set -x

# curl can leave a partial archive after an error. Without the rm call, the test for the file then skips the download

if [ ! -f testmodel.tar.xz ]; then curl --fail --retry 3 -O -L https://github.com/artis-mcrt/artistools/releases/download/v2026.5.9/testmodel.tar.xz || rm -f testmodel.tar.xz; fi

rm -rf testmodel/
mkdir -p testmodel/
tar -xf testmodel.tar.xz --directory testmodel/
# find testmodel -size +1M -exec xz -v {} \;

if [ ! -f vspecpolmodel.tar.xz ]; then curl --fail --retry 3 -O -L https://github.com/artis-mcrt/artistools/releases/download/v2026.5.9/vspecpolmodel.tar.xz || rm -f vspecpolmodel.tar.xz; fi
tar -xf vspecpolmodel.tar.xz

if [ ! -f test_classicmode_3d.tar.xz ]; then curl --fail --retry 3 -O -L https://github.com/artis-mcrt/artistools/releases/download/v2026.5.9/test_classicmode_3d.tar.xz || rm -f test_classicmode_3d.tar.xz; fi
tar -xf test_classicmode_3d.tar.xz

# git holds this archive, because it is small. Thus it needs no download
tar -xf test_classicmode_1d.tar.xz


cp grid.out testmodel/

xz -d testmodel/transitiondata.txt.xz

set +x
