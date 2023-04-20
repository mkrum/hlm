#! /bin/bash

docker run \
    -v $(pwd)/main.hs:/home/ubuntu/main.hs \
    htorch/hasktorch-jupyter \
    /bin/bash -c "ghc main.hs && ./main" \
