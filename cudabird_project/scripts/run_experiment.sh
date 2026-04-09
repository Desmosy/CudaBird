#!/bin/bash
#SBATCH --export=/usr/local/cuda/bin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

cd "${PROJECT_ROOT}" || exit 1

echo "********************************"
echo "** Building CudaBird Project  **"
echo "********************************"

cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j "$(nproc)"

if [ "${RUN_SMOKE_TESTS:-0}" = "1" ]; then
    echo ""
    echo "********************************"
    echo "** Running Smoke Tests        **"
    echo "********************************"
    ctest --test-dir "${BUILD_DIR}" --output-on-failure
fi

echo ""
echo "********************************"
echo "** Running CudaBird Training  **"
echo "********************************"

"${BUILD_DIR}/cudabird" "$@"
