#!/bin/bash
INSTALL_DIR=/usr/local/bin
VERSION="0.5.2"
(
  cd "${INSTALL_DIR}" \
  && curl -L "https://github.com/grailbio/bazel-compilation-database/archive/${VERSION}.tar.gz" | tar -xz \
  && ln -f -s "${INSTALL_DIR}/bazel-compilation-database-${VERSION}/generate.py" bazel-compdb
)
echo "Type bazel-compdb to rebuild database"
