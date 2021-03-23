#!/usr/bin/env zsh

set -x

if [ ! -f testmodel.tar.xz ]; then curl -O https://psweb.mp.qub.ac.uk/artis/artistools/testmodel.tar.xz; fi

mkdir -p artismodel/
tar -xf testmodel.tar.xz --directory artismodel/

set +x