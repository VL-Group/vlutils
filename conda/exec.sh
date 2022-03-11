#!/bin/bash

set -ex
set -o pipefail


conda build -c conda-forge -c bioconda -c xiaosu-zhu --output-folder . .

anaconda upload --label main noarch/*.tar.bz2
