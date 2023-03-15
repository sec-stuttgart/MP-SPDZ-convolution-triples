# MP-SPDZ With Convolution Triples

The original README can be found in [MP-SPDZ.README](MP-SPDZ.README.md).

This repository contains our updates to [MP-SPDZ](https://github.com/data61/MP-SPDZ) to natively support convolutions in MPC programs.
This includes the generation of convolution triples in an offline phase and the use of convolution triples in the online phase.
We also added support for matrix triples in the online phase to compare the native use of convolutions to matrix-based protocols.

Our implementation adds new convolution and matrix multiplication opcodes/instructions to MP-SPDZ's VM-based implementation.
Currently, you have to call these low-level instructions yourself in `.mpc` scripts.
We did not replace high-level functionality (operator overloading or convolution layers for MP-SPDZ's ML library).
This way, both old and new operations are available and can be compared.

Comparing the classical Beaver triple-based approach, the matrix triple-based approach, and our convolution triple-based approach is the main purpose of this repository.
Therefore, we include multiple benchmarks and an automated way to run them.

## Paper

For academic purposes, please cite our paper:

    @article{RiviniusReisertHaslerKuesters-POPETS-2023,
        author    = {Marc Rivinius and Pascal Reisert and Sebastian
                  Hasler and Ralf K{\"{u}}sters},
        title     = {{Convolutions in Overdrive: Maliciously Secure
                  Convolutions for MPC}},
        journal   = {Proc. Priv. Enhancing Technol.},
        volume    = {2023},
        number    = {3},
        year      = {2023},
        note      = {To appear}
    }

We will update this as soon as a DOI is available.
An [eprint](https://eprint.iacr.org/) version will be available soon, as well.

## Build

We tested our implementation with docker-based setups.
For this, first build the docker image which is used to compile and run the programs (this might take a while):

    sudo docker build --tag mpspdz:v0.3.2-convolutions --tag mpspdz:convolutions --target benchmarks .

The resulting image will be around 4 GB in size and contain enryption key and generated (convolution and matrix) triples to run the benchmarks.

## Benchmarks

### Full Benchmarking

The easiest way to replicate our results is to simply run all benchmarks with the pre-configured scripts (this might take several days)

    sudo docker run --rm --cap-add=NET_ADMIN --mount type=bind,source="$(pwd)/benchmarks",target=/usr/src/MP-SPDZ/benchmarks mpspdz:convolutions

(mounting the `benchmarks` directory makes sure that the benchmark results are available on you machine after the container exits).

### Partial Benchmarking

Alternatively, you can start the docker container interactively with

    sudo docker run -it --rm --cap-add=NET_ADMIN --mount type=bind,source="$(pwd)/benchmarks",target=/usr/src/MP-SPDZ/benchmarks mpspdz:convolutions bash

and run individual benchmarks.
For this, make sure to emulate the network settings from our paper to get comparable results, i.e., for the LAN setup:

    export BENCHMARK_DELAY=-10
    tc qdisc add dev lo root handle 1:0 netem delay 10ms rate 1Gbit

or for the WAN setup:

    export BENCHMARK_DELAY=-35
    tc qdisc add dev lo root handle 1:0 netem delay 35ms rate 320Mbit

Then, run the benchmarks:

    sh 1-lowgear.sh
    sh 1-online.sh
    sh 1-highgear.sh

or even smaller parts like

    sh 1.1.lo-benchmark-conv2d-7x7x.sh

which runs the 7x7 convolution of ResNet50 with the LowGear protocol.
The partial scripts follow the naming convention `1.<part>.<protocol>-benchmark-<name>.sh` where `<protocol>` is either `lo` (LowGear-style protocols), `on` (only the online phase), or `hi` (HighGear-style protocols); `<name>` is a short description of what is benchmarked in the `<part>`th part of the benchmark, e.g., emulated 3x3 convolution in part 3 (here, "emulated" means that we do not benchmark the conv2d operations as in ResNet50 but a smaller related convolution as the full conv2d operation would take very long and/or cannot be run in MP-SPDZ without adding support for larger tensors).

Like this, you can run multiple separate benchmarks at once to reduce the overall time it takes to complete all experiments.
Make sure to emulate a network setting whenever you start the containers like this and also make sure to have enough resources (CPU cores and RAM) so concurrent experiments don't influence the performance of each other.
We used 4 cores for 2-party experiments (LowGear and some of the online experiments) and 8 cores for 4-party experiments (HighGear and the rest of the online experiments).

### Benchmark Results

Results are saved by default in the `benchmarks` directory.
Subdirectories correspond to different individual experiments.
Subdirectories with the `.partial` postfix correspond to intermediate results that can be continued from (with some manual work: add `--checkpoint path/to/partial/result.json` for a call to `./Scripts/benchmark.py` that you want to resume from an intermediate result).
Currently, we do not use an automated process to read the results but manually extract the runtime and communication cost from the end of the generated JSON files.

Results without postfix correspond to the "Simple Packing" (e.g., `conv2d` and `lowgear`) except for depthwise convolutions (e.g., `depthwise-conv2d` and `lowgear`; this corresponds to the "Depthwise Packing").
The `-direct` postfix (e.g., `highgear-direct`) indicates the "Generalized Huang et al. Packing".

## Misc.

The script `Scripts/conv2-matmul.py` can be used to count the matrix multiplications needed for CNNs.
Use it, for example, like

    python3 Scripts/conv2-matmul.py resnet50-v1-7.onnx --summary

after downloading the corresponding [model](https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet50-v1-7.onnx) in `.onnx` format.

For depthwise convolutions, use `Scrips/depthwise-conv2-matmul.py` like

    python3 Scripts/depthwise-conv2-matmul.py 7 7 512

to compute the number of matrix multiplications needed for a depthwise convolution of a 7x7 image with depth 512.

## Acknowledgments

Marc Rivinius, Pascal Reisert, and Ralf Küsters were supported by the [CRYPTECS project](https://www.cryptecs.eu/).
The CRYPTECS project has received funding from the German Federal Ministry of Education and Research under Grant Agreement No. 16KIS1441 and from the French National Research Agency under Grant Agreement No. ANR-20-CYAL-0006.
Sebastian Hasler was supported by Advantest as part of the [Graduate School "Intelligent Methods for Test and Reliability" (GS-IMTR)](https://www.gs-imtr.uni-stuttgart.de/) at the University of Stuttgart.
The authors also acknowledge support by the state of Baden-Württemberg through [bwHPC](https://www.bwhpc.de/).
