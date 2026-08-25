# CS633 – Parallel Computing (Coursework)

This repo has my (Group 38's) assignments for **CS633: Parallel Computing**. Both assignments are written in **C using MPI** (Message Passing Interface) — a library that lets many separate processes, possibly on different machines, talk to each other while solving one problem together. Programs were run on **PARAM Rudra**, a supercomputer cluster, using the **SLURM** job scheduler.

In plain words: normally one program runs on one CPU core. With MPI, we start the *same* program on many cores at once (called "ranks" or "processes"), give each one a small piece of the work, and have them send messages to each other (instead of sharing memory) to combine their results.

---

## Assignment 1 — Pairwise message passing (`assignment1/`)

**Goal:** Get comfortable with basic MPI communication (`MPI_Send` / `MPI_Recv` only — no shortcuts allowed).

**What the program does, simply:**
- Every process starts with an array of `M` random numbers.
- Processes are paired up with a partner that is `D1` ranks away, and separately with a partner `D2` ranks away.
- One side of each pair sends its array, the other side does some math on it (squares the numbers for the `D1` pair, takes the log for the `D2` pair) and sends it back.
- This happens for `T` rounds, with senders/receivers swapping roles each round so data flows both ways.
- At the end, every process finds the biggest number it ended up with, and rank 0 (the "leader" process) collects everyone's answer and prints the overall biggest number for `D1` and `D2`, plus how long it all took.

**Files:**
- `src.c` — the MPI program
- `jobScripts/submit*.sh` — SLURM scripts that run it with different process counts (8, 16, 32) and data sizes on the cluster
- `plot.py` — turns `timing.txt` into a boxplot of run times
- `timing.txt` — recorded run times from the cluster
- `Group38.pdf` — our short report explaining the approach and results

**Run it locally:**
```bash
mpicc -o src src.c -lm
mpirun --oversubscribe -np 32 ./src 1048576 2 4 10 1000
# args: M(array size) D1 D2 T(iterations) seed
```

---

## Assignment 2 — 3D stencil computation with halo exchange (`assignment2/`)

**Goal:** A bigger, more realistic parallel program — simulate a value spreading across a 3D grid (like heat or pressure diffusing through space), which is a common pattern in real scientific/HPC software.

**What the program does, simply:**
- Imagine a big 3D box of numbers (`nx × ny × nz` per process). The box is split into smaller blocks, and each MPI process owns one block, arranged in a `px × py × pz` grid of processes.
- Each block only "knows" its own numbers, but the math needs a bit of its *neighbors'* edge data too (like knowing the temperature just outside your room to guess how yours will change). So before each round, neighboring processes exchange these edge/boundary values — this is called **halo exchange**.
- After exchanging, every process updates each of its points to the average of itself and its neighbors (a "stencil" computation — like a blur/smoothing operation).
- It also counts how many times the data crosses a chosen threshold value (`isovalue`), which is useful for finding surfaces/contours in scientific data.
- This repeats for `T` steps, and the leader process (rank 0) collects the total counts and prints the total run time.

**Optimizations we made** (see `Group38.pdf` in `assignment1/` for the write-up style, and the report in this folder):
- Used **non-blocking sends/receives** (`MPI_Isend`/`MPI_Irecv`) so processes can keep working while messages are in flight, instead of waiting.
- Used **derived MPI datatypes** (`MPI_Type_vector`) to send a whole slice of the 3D block in one message, instead of copying data into a temporary buffer first.
- **Swapped buffer pointers** each step instead of copying the whole grid, saving memory and time.
- Result: about **3x faster** than our first naive version (e.g. ~1.5s → ~0.5s at 32 processes on a 120³ grid).

**Files:**
- `src.c` — the MPI program
- `submit.sh` — SLURM script that runs all test configurations (32/48/64/96 processes, grid sizes 120³ and 240³) on the cluster
- `plotScript.py` — plots `timing.txt` as boxplots
- `timing.txt` — recorded run times from the cluster
- `readme.pdf` — detailed write-up of the code/approach
- `job output/` — raw SLURM output/error logs from the cluster runs

**Run it locally:**
```bash
mpicc -o assignment2 src.c -lm
mpirun --oversubscribe -np 32 ./assignment2 7 32 4 4 2 120 120 120 5 1000 2 500
# args: d(stencil size) ppn px py pz nx ny nz T seed F(num fields) isovalue
```

---

## Quick glossary (for anyone new to this)

| Term | Simple meaning |
|---|---|
| **Rank / process** | One copy of the program running; each has a unique ID number starting at 0 |
| **MPI_Send / MPI_Recv** | Basic "blocking" send/receive — the process waits until the message is fully sent/received |
| **MPI_Isend / MPI_Irecv** | "Non-blocking" versions — start the send/receive and keep doing other work, check back later |
| **Halo exchange** | Neighboring processes swapping their boundary data so each has what it needs to compute correctly |
| **Derived datatype** | A custom MPI description of a non-contiguous chunk of memory (e.g. a 2D slice out of a 3D array), so it can be sent in one call |
| **MPI_Reduce** | Combine one value from every process into a single result (e.g. sum, max) at one "root" process |
| **Stencil computation** | Updating each point in a grid using its neighboring points (common in simulations) |
| **Strong scaling** | How much faster a fixed-size problem runs as you add more processes |
