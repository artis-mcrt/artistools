#!/usr/bin/env zsh

set -x

if [ ! -f testmodel.tar.xz ]; then curl -O https://theory.gsi.de/~lshingle/artis_http_public/artistools/testmodel.tar.xz; fi

rm -rf testmodel/
mkdir -p testmodel/
tar -xf testmodel.tar.xz --directory testmodel/
# find testmodel -size +1M -exec xz -v {} \;

if [ ! -f vspecpolmodel.tar.xz ]; then curl -O https://theory.gsi.de/~lshingle/artis_http_public/artistools/vspecpolmodel.tar.xz; fi
tar -xf vspecpolmodel.tar.xz

if [ ! -f test_classicmode_3d.tar.xz ]; then curl -O https://theory.gsi.de/~lshingle/artis_http_public/artistools/test_classicmode_3d.tar.xz; fi
tar -xf test_classicmode_3d.tar.xz


cp grid.out testmodel/

xz -d testmodel/transitiondata.txt.xz

set +x
