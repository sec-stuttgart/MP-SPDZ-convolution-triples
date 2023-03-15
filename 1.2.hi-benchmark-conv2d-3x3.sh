./Scripts/benchmark.py 56 64 3 3 --parties 4 --output_folder "$MP_SPDZ_BENCHMARKS/benchmarks-3x3-4hi-c3$BENCHMARK_DELAY" --approaches "[('conv2d','highgear'),('conv2d','highgear-direct')]" --repeats 2
./Scripts/benchmark.py 28 128 4 3 --parties 4 --output_folder "$MP_SPDZ_BENCHMARKS/benchmarks-3x3-4hi-c4$BENCHMARK_DELAY" --approaches "[('conv2d','highgear'),('conv2d','highgear-direct')]" --repeats 2
./Scripts/benchmark.py 14 256 6 3 --parties 4 --output_folder "$MP_SPDZ_BENCHMARKS/benchmarks-3x3-4hi-c6$BENCHMARK_DELAY" --approaches "[('conv2d','highgear'),('conv2d','highgear-direct')]" --repeats 2

