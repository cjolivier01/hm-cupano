#!/bin/bash
if [ -z "$1" ]; then
  echo "Usage: $(basename $0) <directory>"
  echo ""
  echo "directory:   Directory of the videos you passed to scripts/create_control_pionts.py"
  echo "             or just the name of that directory under ${HOME}/Videos"
  exit 1
fi
./bazel-bin/tests/test_cuda_blend $@
