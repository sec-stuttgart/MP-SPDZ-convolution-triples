./Scripts/benchmark.py 7 512 1 3 256 --parties 4 --output_folder "$MP_SPDZ_BENCHMARKS/benchmarks-3x3-4hi-emul0$BENCHMARK_DELAY" --approaches "[('conv2d','highgear'),('conv2d','highgear-direct')]" --repeats 2
./Scripts/benchmark.py [56,28,14,7] [64,128,256,512] 1 3 2 --zip_sizes_and_depths True --parties 4 --output_folder "$MP_SPDZ_BENCHMARKS/benchmarks-3x3-4hi-emul1$BENCHMARK_DELAY" --approaches "[('base','highgear')]" --repeats 2
