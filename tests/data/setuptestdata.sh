#!/usr/bin/env zsh

set -x

if [ ! -f testmodel.tar.xz ]; then curl -O https://theory.gsi.de/~lshingle/artis_http_public/artistools/testmodel.tar.xz; fi

rm -rf testmodel/
mkdir -p testmodel/
tar -xf testmodel.tar.xz --directory testmodel/
# find testmodel -size +1M -exec xz -v {} \;

set +x