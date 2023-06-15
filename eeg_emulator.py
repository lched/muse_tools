import multiprocessing as mp
import time
from argparse import ArgumentParser

import numpy as np
import pylsl
import pyxdf


def launch_events_stream(stream):
    stream_info = stream["info"]
    sample_rate = float(stream_info["nominal_srate"][0])
    info = pylsl.StreamInfo(
        stream_info["name"][0],
        type=stream_info["type"][0],
        channel_count=int(stream_info["channel_count"][0]),
        nominal_srate=sample_rate,
    )
    outlet = pylsl.StreamOutlet(info)
    wait_times = np.diff(stream["time_stamps"])
    for t, sample in zip(wait_times, stream["time_series"]):
        time.sleep(t)
        outlet.push_sample(list(sample))


def launch_sampled_stream(stream):
    stream_info = stream["info"]
    sample_rate = float(stream_info["nominal_srate"][0])
    info = pylsl.StreamInfo(
        stream_info["name"][0],
        type=stream_info["type"][0],
        channel_count=int(stream_info["channel_count"][0]),
        nominal_srate=sample_rate,
    )
    outlet = pylsl.StreamOutlet(info)
    for sample in stream["time_series"]:
        time.sleep(1 / sample_rate)
        outlet.push_sample(list(sample))


def main(xdf_file):
    # Load XDF file and get data
    streams, _ = pyxdf.load_xdf(xdf_file)

    pool = mp.Pool(processes=len(streams))

    for stream in streams:
        if float(stream["info"]["nominal_srate"][0]) == 0:
            task = launch_events_stream
        else:
            task = launch_sampled_stream
        pool.apply_async(task, args=(stream,))
    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("xdf_file", type=str)
    args, _ = parser.parse_known_args()
    main(args.xdf_file)
