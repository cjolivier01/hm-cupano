#!/bin/bash
set -e
cd /tmp
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install .
