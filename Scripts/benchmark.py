#!/usr/bin/env python3

from datetime import datetime
import json
import shlex
import subprocess
import os
import sys
import tempfile
from tqdm import tqdm as progress_bar

DIAGONAL = "DIAGONAL"

def executable(what):
    if what == "online":
        return "Player-Online.x"
    else:
        return f"{what}-party.x"

def parse_range_arg(arg):
    if isinstance(arg, tuple):
        return list(range(*arg))
    elif isinstance(arg, list):
        return arg
    else:
        return [arg]

def progress(iterable, **kwargs):
    p = progress_bar(iterable, **kwargs)
    for x in p:
        p.set_postfix({"current": x})
        yield x

def repeatedly(repeats, parallel_repeats):
    if repeats > 1 and parallel_repeats == 1:
        for repeat in progress(range(repeats), desc="repeats", leave=False):
            yield (repeat,)
    elif repeats == 0:
        yield (0,)
    else:
        for repeat in range(0, repeats, parallel_repeats):
            if repeat + parallel_repeats < repeats:
                yield tuple(repeat + i for i in range(parallel_repeats))
            else:
                yield tuple(range(repeat, repeats))


def do_plot(image_sizes, depths, approaches, results_time, results_data, zip_sizes_and_depths, experiment_title, run_title, plot="plot.json", figsize=(16/6,9/2)):
    from matplotlib import pyplot as plt

    units = {"time": "s", "data": "MB"}
    dims = size_and_depth_dimensions(image_sizes, depths, zip_sizes_and_depths)

    figsize = figsize[0] * dims[1] * len(approaches), figsize[1] * dims[0]

    for what, Y in zip(("time", "data"), (results_time, results_data)):
        fig, axes = plt.subplots(dims[0], dims[1], figsize=figsize)
        for (i, image_size), (j, depth) in enumerate_size_and_depth(image_sizes, depths, zip_sizes_and_depths):
            if dims[0] == 1 and dims[1] == 1:
                ax = axes
            elif dims[0] == 1:
                ax = axes[j]
            elif dims[1] == 1:
                ax = axes[i]
            else:
                ax = axes[i,j]
            Y_i = [Y[image_size, depth, approach, engine] for approach, engine in approaches]
            ax.bar(range(len(approaches)), Y_i)
            ax.set_ylim(0, None)
            ax.set_xticks(range(len(approaches)))
            ax.set_xticklabels([run_title.format(approach, engine) for approach, engine in approaches])
            ax.set_xlabel(experiment_title.format(image_size, depth))
            ax.set_ylabel(f"{what} ({units[what]})")

        if zip_sizes_and_depths == DIAGONAL:
            for i in range(dims[0]):
                for j in range(i+1, dims[1]):
                    axes[i,j].set_axis_off()

        fig.tight_layout()
        fig.savefig(f"{plot}.{what}.pdf", format="pdf", bbox_inches="tight")

def to_json(result, image_sizes, depths, approaches, results_time, results_data, zip_sizes_and_depths):
    for image_size, depth in size_and_depth(image_sizes, depths, zip_sizes_and_depths):
        for approach, engine in approaches:
            if (image_size, depth, approach, engine) in results_time:
                assert (image_size, depth, approach, engine) in results_data

                if image_size not in result:
                    result[image_size] = {}
                if depth not in result[image_size]:
                    result[image_size][depth] = {}
                if approach not in result[image_size][depth]:
                    result[image_size][depth][approach] = {}
                assert engine not in result[image_size][depth][approach]

                result[image_size][depth][approach][engine] = { "data": results_data[image_size, depth, approach, engine], "time": results_time[image_size, depth, approach, engine] }


def from_json(results_time, results_data, image_sizes, depths, approaches, result, zip_sizes_and_depths):
    for image_size, depth in size_and_depth(image_sizes, depths, zip_sizes_and_depths):
        for approach, engine in approaches:
            try:
                time = result[str(image_size)][str(depth)][str(approach)][str(engine)]["time"]
                data = result[str(image_size)][str(depth)][str(approach)][str(engine)]["data"]
                results_time[image_size, depth, approach, engine] = time
                results_data[image_size, depth, approach, engine] = data
            except KeyError:
                pass

def load_json(file, results_time, results_data, image_sizes, depths, approaches, zip_sizes_and_depths, logs):
    with open(file) as f:
        result = json.loads(f.read())
    if result["zip_sizes_and_depths"] == DIAGONAL:
        assert zip_sizes_and_depths in (DIAGONAL, False)
    elif result["zip_sizes_and_depths"] == True:
        assert zip_sizes_and_depths == True
    try:
        for log in result["logs"]:
            logs.append(log)
    except KeyError:
        pass
    from_json(results_time, results_data, image_sizes, depths, approaches, result, zip_sizes_and_depths)

def save_json(file, image_sizes, depths, approaches, results_time, results_data, program, count, args, parties, repeats, zip_sizes_and_depths, logs):
    result = { "argv" : shlex.join(sys.argv), "program": program, "count": count, "args" : args, "parties": parties, "repeats": repeats, "image_sizes" : image_sizes, "depths" : depths, "approaches" : approaches, "zip_sizes_and_depths" : zip_sizes_and_depths, "logs" : logs }
    to_json(result, image_sizes, depths, approaches, results_time, results_data, zip_sizes_and_depths)
    with open(file, "w") as f:
        f.write(json.dumps(result, indent=4))

def progress_size_and_depth(image_sizes, depths, zip_sizes_and_depths):
    if zip_sizes_and_depths == True:
        assert len(image_sizes) == len(depths)
        yield from progress(list(zip(image_sizes, depths)), desc="image_size,depth", leave=True)
    elif zip_sizes_and_depths == DIAGONAL:
        assert len(image_sizes) == len(depths)
        for i, image_size in enumerate(progress(image_sizes, desc="image_size", leave=True)):
            for depth in progress(depths[:i+1], desc="depth", leave=False):
                yield image_size, depth
    else:
        for image_size in progress(image_sizes, desc="image_size", leave=True):
            for depth in progress(depths, desc="depth", leave=False):
                yield image_size, depth

def size_and_depth(image_sizes, depths, zip_sizes_and_depths):
    if zip_sizes_and_depths == True:
        assert len(image_sizes) == len(depths)
        yield from zip(image_sizes, depths)
    elif zip_sizes_and_depths == DIAGONAL:
        assert len(image_sizes) == len(depths)
        for i, image_size in enumerate(image_sizes):
            for depth in depths[:i+1]:
                yield image_size, depth
    else:
        for image_size in image_sizes:
            for depth in depths:
                yield image_size, depth

def enumerate_size_and_depth(image_sizes, depths, zip_sizes_and_depths):
    if zip_sizes_and_depths == True:
        assert len(image_sizes) == len(depths)
        yield from zip(enumerate(image_sizes), enumerate(depths))
    elif zip_sizes_and_depths == DIAGONAL:
        assert len(image_sizes) == len(depths)
        for i, image_size in enumerate(image_sizes):
            for j, depth in enumerate(depths[:i+1]):
                yield (i, image_size), (j, depth)
    else:
        for i, image_size in enumerate(image_sizes):
            for j, depth in enumerate(depths):
                yield (i, image_size), (j, depth)

def size_and_depth_dimensions(image_sizes, depths, zip_sizes_and_depths):
    if zip_sizes_and_depths == True:
        assert len(image_sizes) == len(depths)
        return len(image_sizes), 1
    else:
        return len(image_sizes), len(depths)

def main(image_sizes, depths, count, *args, program="benchmark_conv2d", executable_folder=".", parties=2, port_offset=0, repeats=1, parallel_repeats=False, approaches=(("base", "cowgear"), ("conv2d", "cowgear"), ("base", "chaigear"), ("conv2d", "chaigear")), output_folder="benchmarks", checkpoints_folder=None, zip_sizes_and_depths=False, plot=False, checkpoint=True, clear_checkpoint=False, experiment_title="w=h={}, depth={}", run_title="{}@{}"):

    image_sizes = parse_range_arg(image_sizes)
    depths = parse_range_arg(depths)

    approach_names = [approach for approach, engine in approaches]
    engine_names = [engine for approach, engine in approaches]
    assert all(os.path.exists(os.path.join(executable_folder, executable(engine))) for engine in engine_names)

    results_time = {}
    results_data = {}
    logs = []

    if isinstance(zip_sizes_and_depths, str):
        zip_sizes_and_depths = zip_sizes_and_depths.upper()
    assert zip_sizes_and_depths in (DIAGONAL, True, False)

    if isinstance(plot, str):
        load_json(plot, results_time, results_data, image_sizes, depths, approaches, zip_sizes_and_depths, logs)

        do_plot(image_sizes, depths, approaches, results_time, results_data, zip_sizes_and_depths, plot=plot, experiment_title=experiment_title, run_title=run_title)
        return

    if isinstance(parallel_repeats, bool):
        if parallel_repeats:
            parallel_repeats = repeats
        else:
            parallel_repeats = 1

    if checkpoints_folder is None:
        checkpoints_folder = f"{output_folder}.partial"
    os.makedirs(checkpoints_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    now = datetime.now()
    now = f"{now:%Y-%m-%d_%H-%M-%S}"
    file_base = f"{program}_{now}_{image_sizes[0]}+{len(image_sizes)}_{depths[0]}+{len(depths)}_{count}_{'-'.join(map(str, args))}_+{len(approaches)}"
    json_file = f"{file_base}.json"
    json_file = os.path.join(output_folder, json_file)
    if isinstance(checkpoint, str):
        load_json(checkpoint, results_time, results_data, image_sizes, depths, approaches, zip_sizes_and_depths, logs)
    elif checkpoint:
        checkpoint = f"{file_base}.partial.json"
        checkpoint = os.path.join(checkpoints_folder, checkpoint)


    if not os.path.exists("Player-Data/2-p-128/Params-Data"):
        subprocess.check_output(["./Scripts/setup-online.sh", str(parties)])

    for image_size, depth in progress_size_and_depth(image_sizes, depths, zip_sizes_and_depths):
        for approach in progress(approach_names, desc="approach(compile)", leave=False):
            compile_call = ["./compile.py", program, approach, str(image_size), str(depth), str(count)] + list(map(str, args))
            subprocess.check_output(compile_call)

        for approach, engine in progress(approaches, desc="approach,engine", leave=False):
            mpc_file = f"{program}-{approach}-{image_size}-{depth}-{count}"
            for arg in args:
                mpc_file += f"-{arg}"

            if (image_size, depth, approach, engine) in results_time:
                assert (image_size, depth, approach, engine) in results_data
                continue

            timings = []
            datasum = []

            process_args = [[[os.path.join(executable_folder, executable(engine)), "-p", str(i), "-N", str(parties), "-pn", str(5000 + port_offset + 10 * parties * parties * repeat), mpc_file] for i in range(parties)] for repeat in range(repeats)]
            processes = [list() for _ in range(repeats)]
            tmpfiles = [list() for _ in range(repeats)]
            for repeat_block in repeatedly(repeats, parallel_repeats):
                for repeat in repeat_block:
                    for i in range(parties):
                        outf = tempfile.TemporaryFile()
                        errf = tempfile.TemporaryFile()
                        process = subprocess.Popen(process_args[repeat][i], stdin=subprocess.DEVNULL, stdout=outf, stderr=errf)
                        processes[repeat].append(process)
                        tmpfiles[repeat].append((outf, errf))

                for repeat in repeat_block:
                    for i, process in enumerate(processes[repeat]):
                        code = process.wait()
                        outf, errf = tmpfiles[repeat][i]
                        outf.seek(0)
                        output = outf.read()
                        outf.close()
                        errf.seek(0)
                        error = errf.read()
                        errf.close()

                        if code is None or code != 0:
                            print("error code:", code, "\nerror:", error, "\noutput:", output)
                            raise subprocess.CalledProcessError(code, process_args[repeat][i], output=output, stderr=error)

                        lines = error.decode().splitlines()
                        time_line = next(line for line in lines if line.startswith("Time = "))
                        time = float(time_line.split()[2])
                        data_line = next(line for line in lines if line.startswith("Data sent = "))
                        data = float(data_line.split()[3])

                        timings.append(time)
                        datasum.append(data)

                        logs.append({
                            "output" : output.decode(),
                            "error" : error.decode(),
                            "repeat" : repeat,
                            "party" : i,
                            "args" : process_args[repeat][i],
                            "parameters" : {
                                "image_size" : image_size,
                                "depth" : depth,
                                "approach" : approach,
                                "engine" : engine,
                                "count" : count,
                                "args" : args
                            },
                            "time" : time,
                            "data" : data,
                            "finished_at" : str(datetime.now()),
                        })

            results_time[image_size, depth, approach, engine] = sum(timings) / len(timings)
            results_data[image_size, depth, approach, engine] = sum(datasum) / len(datasum)
            if checkpoint:
                save_json(checkpoint, image_sizes, depths, approaches, results_time, results_data, program, count, args, parties, repeats, zip_sizes_and_depths, logs)

    save_json(json_file, image_sizes, depths, approaches, results_time, results_data, program, count, args, parties, repeats, zip_sizes_and_depths, logs)
    if checkpoint and clear_checkpoint:
        os.remove(checkpoint)

    if plot:
        do_plot(image_sizes, depths, approaches, results_time, results_data, zip_sizes_and_depths, plot=json_file, experiment_title=experiment_title, run_title=run_title)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
