# CudaBird

CudaBird is a CUDA-based Flappy Bird training project that evolves many small neural networks in parallel on the GPU.

### Runtime and tooling

The current branch builds and runs successfully with:

```bash
cmake -S cudabird_project -B cudabird_project/build -DCMAKE_BUILD_TYPE=Release
cmake --build cudabird_project/build -j 4
ctest --test-dir cudabird_project/build --output-on-failure
cd cudabird_project && ./build/cudabird --population 512 --generations 3 --ticks 512 --seed 123
```

## Example Output

```text
CudaBird training run
Population: 512 | generations: 3 | max ticks/gen: 512 | report every: 1 | seed: 123
Generation    0 | best fitness   502.00 | avg fitness    49.10 | best score    1 | avg score   0.00
Generation    1 | best fitness  1182.00 | avg fitness    77.44 | best score    3 | avg score   0.02
Generation    2 | best fitness   502.00 | avg fitness   109.54 | best score    1 | avg score   0.01
Best run: generation 1, fitness 1182.00, score 3
```
