# Random Quantum Codes

This repository contains the work for my computer science senior thesis capstone at CU Boulder in the spring semester of 2023 along with some further experiments involving random quantum codes. This is all still a work-in-progress.

This all was made with help from
 - Dr. Josh Combes
 - Aiko Kyle
 - Noah Lordi
 - Steven Liu

## Codebase Structure

Project dependencies are encoded in requirements.txt, and experiments are done in the top-level Jupyter Notebooks. The main code is stored in the `src` folder. Running any of the code may generate files in a top-level folder called `data`. The data folder stores cached results so that already-done computations can be read rather than re-calculated. Non-random code encodings are cached along with their fidelities when calculated. The supermatrix representations of noise channels are also stored when computed. Random code information is also automatically saved, but only if the random code has a better fidelity in its conditions (code parameters, noise channel, etc.) than whatever is already stored -- if anything -- for those conditions. This means that for any specific set of conditions, only information for the best random code is automatically kept, and when a random code is determined to be better than whatever is currently stored, whatever is currently stored gets overwritten. Importantly for random codes, the number of used/checked random codes in any set of conditions is not saved.

The code and its file-IO is generally multiprocess-friendly with some caveats. For non-random codes, calculations can be done in parallel as long as multiple calculations with the same conditions aren't being done at the same time; each parallel calculation must have unique conditions. This is a reasonable constraint for non-random codes because the same conditions should yield the same results. However, for random codes, it is sensible -- or even desired -- to use the same conditions for multiple calculations. If running multiple calculations with the same conditions for random codes, a lock should be passed in to the `code_simulator.calculate_code_fidelity` function for its optional `random_code_file_io_mutex` parameter. The lock is used to ensure that the best random code is tracked accurately and that the file-IO is done without race conditions.

## Comparing Codes

Comparing encoding schemes is difficult. The three ways comparison is facilitated in this codebase are
1. Generating Wigner plots with `code_simulator.make_wigner_plots_for`.
2. Visually comparing fidelity distributions as done in `RandomCodeComparison.ipynb` (this only really makes sense for random codes).
3. Computing dot products squared of the logical states of encodings, as given by `code_simulator.compute_code_similarities` (this returns 4 numbers, 3 of which will be unique).
Each of these has their strengths and drawbacks, and they're all imperfect ways of determining whether two encoding schemes are similar or dissimilar.
