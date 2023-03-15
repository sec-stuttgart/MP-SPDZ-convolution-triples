## requirements to build MP-SPDZ
FROM ubuntu:22.04 as buildenv

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && apt-get install -y --no-install-recommends \
                automake \
                build-essential \
                clang-11 \
                git \
                iproute2 \
                libboost-dev \
                libboost-thread-dev \
                libclang-dev \
                libgf2x-dev \
                libgmp-dev \
                libntl-dev \
                libsodium-dev \
                libssl-dev \
                libtool \
                m4 \
                texinfo \
                python3-pip \
                yasm \
                vim \
                gdb \
                valgrind \
        && rm -rf /var/lib/apt/lists/*

# mpir
COPY --from=initc3/mpir:55fe6a9 /usr/local/mpir/include/* /usr/local/include/
COPY --from=initc3/mpir:55fe6a9 /usr/local/mpir/lib/* /usr/local/lib/
COPY --from=initc3/mpir:55fe6a9 /usr/local/mpir/share/info/* /usr/local/share/info/

ARG mp_spdz_home="/usr/src/MP-SPDZ"
ENV MP_SPDZ_HOME ${mp_spdz_home}
WORKDIR ${MP_SPDZ_HOME}

RUN pip install --upgrade pip ipython

COPY . .

RUN pip install -r requirements.txt


## building the executables
FROM buildenv as build-benchmarks

ARG mp_spdz_benchmarks="${mp_spdz_home}/benchmarks"
ENV MP_SPDZ_BENCHMARKS ${mp_spdz_benchmarks}

RUN mkdir -p "$MP_SPDZ_BENCHMARKS"
RUN python3 build.py


## copy over the executables in a container without the build requirements
FROM ubuntu:22.04 as benchmarks

LABEL org.opencontainers.image.source=https://github.com/sec-stuttgart/MP-SPDZ-convolution-triples

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && apt-get install -y --no-install-recommends \
                iproute2 \
                libgf2x-dev \
                python3-pip \
        && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip ipython

ARG mp_spdz_home="/usr/src/MP-SPDZ"
ENV MP_SPDZ_HOME ${mp_spdz_home}
WORKDIR ${MP_SPDZ_HOME}

COPY --from=build-benchmarks ${MP_SPDZ_HOME} .

RUN rm -r logs \
 && rm -r Compiler/__pycache__ \
 && rm -r Compiler/GC/__pycache__ \
 && rm -r Programs/Bytecode \
 && rm -r Programs/Schedules \
 && rm -r .git

RUN pip install -r requirements.txt

CMD [ "/bin/bash", "benchmark.sh" ]
