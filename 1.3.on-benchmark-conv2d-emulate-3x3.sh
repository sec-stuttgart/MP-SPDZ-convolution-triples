./Scripts/benchmark.py 7 512 1 3 256 --parties 2 --output_folder "$MP_SPDZ_BENCHMARKS/benchmarks-3x3-2on-emul0$BENCHMARK_DELAY" --approaches "[('base','online'),('conv2d','online'),('matmul','online')]" --repeats 2
./Scripts/benchmark.py 7 512 1 3 256 --parties 4 --output_folder "$MP_SPDZ_BENCHMARKS/benchmarks-3x3-4on-emul0$BENCHMARK_DELAY" --approaches "[('base','online'),('conv2d','online'),('matmul','online')]" --repeats 2
