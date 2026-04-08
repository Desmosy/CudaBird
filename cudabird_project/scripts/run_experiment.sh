#!/bin/bash
#SBATCH --export=/usr/local/cuda/bin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

# Move to the project root directory
cd "$(dirname "$0")/.." || exit

echo "********************************"
echo "** Building CudaBird Project  **"
echo "********************************"

# Rebuild the project if the main executable does not exist
if [ ! -f build/cudabird ]; then
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build . -j $(nproc)
    cd ..
fi

echo ""
echo "********************************"
echo "** Running CudaBird Training  **"
echo "********************************"

# Run the main application
./build/cudabird "$@"

# --- Future: uncomment when tests are enabled in CMakeLists.txt ---
# echo ""
# echo "********************************"
# echo "** Running Unit Tests         **"
# echo "********************************"
# ./build/test_game
# ./build/test_network
# ./build/test_evolution
