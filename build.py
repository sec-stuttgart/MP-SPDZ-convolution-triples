#!/usr/bin/env python3

import subprocess
import os
import shutil

DEFAULT_CONFIG = """ARCH = -march=native
CXX = g++
USE_NTL = 1
MY_CFLAGS += -I/usr/local/include
MY_LDLIBS += -Wl,-rpath -Wl,/usr/local/lib -L/usr/local/lib
PREP_DIR = '-DPREP_DIR="Player-Data/"'
SSL_DIR = '-DSSL_DIR="Player-Data/"'
SECURE = '-DINSECURE'
"""

def write_config(*extra_lines):
    content = DEFAULT_CONFIG
    for line in extra_lines:
        content += f"{line}\n"
    with open("CONFIG.mine", "w") as config:
        config.write(content)

cpus = len(os.sched_getaffinity(0))
def make(*what):
    return subprocess.check_output(["make", f"-j{cpus}", *what])

def run(program, *what, **kwargs):
    return subprocess.check_output([f"./Scripts/{program}.sh", *what], **kwargs)

def build(flags, targets, clean_everything=False):
    if clean_everything:
        make("clean")
    else:
        make("clean-non-x")

    write_config(*flags)
    make(*targets.keys())
    for key, value in targets.items():
        shutil.move(key, value)

os.makedirs("static", exist_ok=True)

builds = [
    # (["GEAR_DIRECT_SUM = 1", "HIGHGEAR_NO_SUMS = 1"], {"static/chaigear-party.x": "chaigear-direct-nosum-party.x", "static/highgear-party.x": "highgear-direct-nosum-party.x"}),
    # (["GEAR_DIRECT_SUM = 1", "HIGHGEAR_GENERIC_EC = 1"], {"static/chaigear-party.x": "chaigear-direct-generic-party.x", "static/highgear-party.x": "highgear-direct-generic-party.x"}),
    # (["GEAR_DIRECT_SUM = 1", "HIGHGEAR_NO_SUMS = 1", "HIGHGEAR_GENERIC_EC = 1"], {"static/chaigear-party.x": "chaigear-direct-nosum-generic-party.x", "static/highgear-party.x": "highgear-direct-nosum-generic-party.x"}),
    (["GEAR_DIRECT_SUM = 1"], 
        {
            # "static/chaigear-party.x": "chaigear-direct-party.x", 
            "static/highgear-party.x": "highgear-direct-party.x",
        }
    ),
    # (["GEAR_BASIC_MATMUL = 1", "HIGHGEAR_NO_SUMS = 1"], {"static/chaigear-party.x": "chaigear-basic-party.x", "static/highgear-party.x": "highgear-basic-party.x"}),
    # (["HIGHGEAR_NO_SUMS = 1"], {"static/chaigear-party.x": "chaigear-nosum-party.x", "static/highgear-party.x": "highgear-nosum-party.x"}),
    # (["HIGHGEAR_GENERIC_EC = 1"], {"static/chaigear-party.x": "chaigear-generic-party.x", "static/highgear-party.x": "highgear-generic-party.x"}),
    # (["HIGHGEAR_NO_SUMS = 1", "HIGHGEAR_GENERIC_EC = 1"], {"static/chaigear-party.x": "chaigear-nosum-generic-party.x", "static/highgear-party.x": "highgear-nosum-generic-party.x"}),
    ([], 
        {
            # "static/chaigear-party.x": "chaigear-party.x",
            "static/highgear-party.x": "highgear-party.x",
        }
    ),

    # (["GEAR_DIRECT_SUM = 1", "LOWGEAR = 1", "LOWGEAR_FILTER_CIPHERTEXTS = 1"], {"static/cowgear-party.x": "cowgear-direct-swapped-party.x", "static/lowgear-party.x": "lowgear-direct-swapped-party.x"}),
    (["GEAR_DIRECT_SUM = 1", "LOWGEAR = 1"], 
        {
            # "static/cowgear-party.x": "cowgear-direct-party.x",
            "static/lowgear-party.x": "lowgear-direct-party.x",
        }
    ),
    # (["GEAR_BASIC_MATMUL = 1", "LOWGEAR = 1", "LOWGEAR_FILTER_CIPHERTEXTS = 1"], {"static/cowgear-party.x": "cowgear-basic-swapped-party.x", "static/lowgear-party.x": "lowgear-basic-swapped-party.x"}),
    # (["GEAR_BASIC_MATMUL = 1", "LOWGEAR = 1"], {"static/cowgear-party.x": "cowgear-basic-party.x", "static/lowgear-party.x": "lowgear-basic-party.x"}),
    # (["LOWGEAR = 1", "LOWGEAR_FILTER_CIPHERTEXTS = 1"], {"static/cowgear-party.x": "cowgear-swapped-party.x", "static/lowgear-party.x": "lowgear-swapped-party.x"}),
    (["LOWGEAR = 1", "LOWGEAR_EXPANDED_BGV = 1", "LOWGEAR_NO_EXPANDED_MASK = 1"],
        {
            # "static/cowgear-party.x": "cowgear-expanded-NOMASK-party.x",
            "static/lowgear-party.x": "lowgear-expanded-NOMASK-party.x",
        }
    ),
    (["LOWGEAR = 1", "LOWGEAR_EXPANDED_BGV = 1"],
        {
            # "static/cowgear-party.x": "cowgear-expanded-party.x",
            "static/lowgear-party.x": "lowgear-expanded-party.x",
        }
    ),
    (["LOWGEAR = 1"],
        {
            "static/Fake-Offline.x": "Fake-Offline.x",
            "static/Player-Online.x": "Player-Online.x",
            # "static/cowgear-party.x": "cowgear-party.x",
            "static/lowgear-party.x": "lowgear-party.x",
        }
    ),
]

make("clean")
print("Building with", cpus, "CPUs (of", os.cpu_count(), "available cores)")
for i, args in enumerate(builds):
    print(f"Building {i+1}/{len(builds)}: {', '.join(map(str, args[1].values()))}", flush=True)
    build(*args)
make("clean-non-x")


print("Setting up preprocessing material")
shutil.rmtree("Player-Data", ignore_errors=True)
count = 1
for n in [2, 4]:
    subprocess.check_output(["./Fake-Offline.x", str(n), "-conv", "1,224,224,3,7,7,32", "-matmul", "12544,147,32", "-d", str(count)])

    subprocess.check_output(["./Fake-Offline.x", str(n), "-conv", "1,56,56,64,3,3,64", "-matmul", "3136,576,64", "-d", str(count)])

    subprocess.check_output(["./Fake-Offline.x", str(n), "-conv", "1,28,28,128,7,7,128", "-matmul", "784,1152,128", "-d", str(count)])

    subprocess.check_output(["./Fake-Offline.x", str(n), "-conv", "1,14,14,256,3,3,256", "-matmul", "196,2304,256", "-d", str(count)])

    subprocess.check_output(["./Fake-Offline.x", str(n), "-conv", "1,7,7,512,3,3,256", "-matmul", "49,4608,256", "-d", str(count)])

    for i in [7,9,11,13,15,17,19,21,23,25,50,120,240]:
        subprocess.check_output(["./Fake-Offline.x", str(n), "-conv", f"1,{i},{i},1,3,3,-1", "-matmul", f"{i * i},9,1", "-d", str(count)])

subprocess.check_output(["./Fake-Offline.x", "2", "-matmul", "128,128,128", "-d", str(count)])

subprocess.check_output(["./compile.py", "tutorial"])
for i in range(4):
    with open(f"Player-Data/Input-P{i}-0", "w") as f:
        f.write("""1 2 3 4
""")

print("Running example with protocols for key generation")
run("run-online", "tutorial")
# run("cowgear", "tutorial")
run("lowgear", "tutorial")
# run("chaigear", "tutorial", env={"PLAYERS" : "4"})
run("highgear", "tutorial", env={"PLAYERS" : "4"})
