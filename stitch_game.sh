#!/bin/bash
GAME_ID="$1"
GAME_DIR="${HOME}/Videos/${GAME_ID}"
./bazel-bin/tests/stitch_two_videos --left="${GAME_DIR}/left.mp4" --right="${GAME_DIR}/right.mp4" --adjust=1 --level=6 --control "${GAME_DIR}" ${@:2}
