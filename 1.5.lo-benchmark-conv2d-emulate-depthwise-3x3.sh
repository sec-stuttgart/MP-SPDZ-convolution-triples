./Scripts/benchmark.py [50,120,240] 32 1 3 --parties 2 --output_folder "$MP_SPDZ_BENCHMARKS/benchmarks-depthwise-3x3-2lo-emul0$BENCHMARK_DELAY" --approaches "[(depthwise-base','lowgear'),('emulate-depthwise-conv2d','lowgear'),('emulate-depthwise-conv2d','lowgear-direct'),('depthwise-conv2d'','lowgear')]" --repeats 2
