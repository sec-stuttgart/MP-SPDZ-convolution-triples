set -e

export BENCHMARK_DELAY=-10
tc qdisc add dev lo root handle 1:0 netem delay 10ms rate 1Gbit
sh 1-lowgear.sh
sh 1-online.sh
sh 1-highgear.sh
tc qdisc del dev lo root # or restart the docker container and then continue below
export BENCHMARK_DELAY=-35
tc qdisc add dev lo root handle 1:0 netem delay 35ms rate 320Mbit
sh 1-lowgear.sh
sh 1-online.sh
sh 1-highgear.sh
