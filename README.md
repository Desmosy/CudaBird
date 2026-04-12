# CudaBird

CudaBird is a CUDA-based Flappy Bird training project that evolves many small neural networks in parallel on the GPU.

## Current Status

The project now has a runnable end-to-end training slice:

- Thousands of birds can be simulated in parallel on the GPU.
- Each bird is controlled by a small feed-forward neural network.
- A generation is scored by survival time and pipes cleared.
- The best-performing birds are used to seed the next generation.
- Mutated offspring are generated directly on the GPU.
- Training metrics are logged each generation.
- The current best network is checkpointed to disk.
- The final champion of a run is exported as a replay video of the evaluated winner episode.

## What Has Been Accomplished

### Core simulation

- Added concrete GPU-friendly game state structures for birds and pipes.
- Implemented CUDA kernels to reset game state for a full population.
- Implemented CUDA kernels to simulate a full generation of Flappy Bird games in parallel.
- Added collision handling, pipe recycling, scoring, and survival-based fitness.

### Neural network layer

- Added a compact neural network representation for each bird.
- Implemented GPU initialization of random network weights using `curand`.
- Implemented forward inference for the bird controller network.

### Evolution loop

- Added generation summary collection on the host.
- Added elite preservation and mutation-based reproduction.
- Added ranking and parent selection for the next generation.

### Runtime and tooling

- Added a runnable training executable in `cudabird_project/src/main.cu`.
- Added CLI options for:
  - `--population`
  - `--generations`
  - `--ticks`
  - `--report-every`
  - `--seed`
- Simplified the CMake build so the project builds as a reusable core library plus executables.
- Added smoke tests for game simulation, network initialization, and evolution.
- Updated the experiment script to rebuild cleanly and optionally run tests.

### Logging and artifacts

- Added console generation summaries during training.
- Added `training_log.csv` output.
- Added checkpoint files for the current best network under `cudabird_project/outputs/`.
- Added OpenCV-based replay export for the actual winning bird episode as `.avi` video.

## Verified Working

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

## Repository Layout

- `cudabird_project/src/game/`: game state and CUDA simulation kernels
- `cudabird_project/src/neural_net/`: network representation and initialization
- `cudabird_project/src/evolution/`: ranking, summaries, and evolution kernels
- `cudabird_project/src/visualization/`: logging and checkpoint output
- `cudabird_project/tests/`: smoke tests
- `cudabird_project/scripts/`: experiment runner

## What Remains

### High-priority next steps

- Add checkpoint loading so training can resume from saved runs.
- Persist full experiment metadata, not just best-network weights.
- Track best-so-far performance across longer runs more clearly.

### Training quality improvements

- Improve parent selection beyond the current simple top-pool sampling.
- Add crossover support or richer mutation strategies.
- Tune fitness shaping to better reward pipe clearing and stability.
- Add normalization or better observation encoding for the controller inputs.
- Support multiple hidden-layer or configurable network shapes.

### Performance and GPU improvements

- Reduce host-device transfers during generation ranking and stats collection.
- Move more of the selection/ranking pipeline onto the GPU.
- Add batch experiment modes for large sweeps.
- Profile occupancy, memory traffic, and kernel runtime.

### Project completeness

- Add stronger unit and regression tests.
- Add CI for build and smoke-test validation.
- Add configuration files or presets for reproducible experiments.
- Add documentation for exported checkpoint format.

## Output Files

After a run you should expect artifacts like:

- `cudabird_project/training_log.csv`
- `cudabird_project/outputs/best_network_latest.txt`
- `cudabird_project/outputs/best_network_gen_XXXX.txt`
- `cudabird_project/outputs/best_replay_latest.avi`
- `cudabird_project/outputs/best_replay_gen_XXXX.avi`
