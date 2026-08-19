#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

if [ "$#" -ne 0 ]; then
    echo "usage: bash ./test.sh" >&2
    exit 2
fi

cd "${repo_dir}"
export PYTHONDONTWRITEBYTECODE=1
exec python3 -m unittest -v \
    tests.test_doctor \
    tests.test_doctor_cli
