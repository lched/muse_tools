""" CAUTION: this only works with *ONE* LSL stream of each type.
To support multiple streams of the same type, we need to pass indexing
to the single_lsl_stream_to_osc I think. (TODO)"""

import multiprocessing as mp
import time
from argparse import ArgumentParser
from threading import Thread, Lock

import numpy as np
import pylsl
from pythonosc import udp_client

# from .utils import get_channels_names

LSL_SCAN_TIMEOUT = 3
LSL_MAX_SAMPLES = 1
LSL_PULL_TIMEOUT = 0.0
OSC_IP = "127.0.0.1"  # Replace with your OSC server IP
OSC_PORT = 9000  # Replace with your OSC server port

# UPDATE RATES (In Hz)
FFT_COMPUTE_RATE = 10
FREQUENCY_BANDS_RATE = 10
CHECK_STREAM_RATE = 10

# FFT parameters
N_FFT = 128


def single_lsl_stream_to_osc(stream_type):
    stream = pylsl.resolve_byprop("type", stream_type, timeout=LSL_SCAN_TIMEOUT)[0]
    # Create LSL inlet
    inlet = pylsl.StreamInlet(stream, max_chunklen=LSL_MAX_SAMPLES)

    # Create an OSC client
    osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
    while True:
        # Read a chunk of samples from LSL
        samples, timestamps = inlet.pull_chunk(
            timeout=LSL_PULL_TIMEOUT, max_samples=LSL_MAX_SAMPLES
        )
        if timestamps:
            # Send raw signals
            osc_client.send_message(f"/muse/{stream_type.lower()}", samples[0])


def eeg_stream_to_osc(stream_type, remove_aux):
    stream = pylsl.resolve_byprop("type", stream_type, timeout=LSL_SCAN_TIMEOUT)[0]
    # Create LSL inlet
    inlet = pylsl.StreamInlet(stream, max_chunklen=LSL_MAX_SAMPLES)
    n_channels = inlet.info().channel_count()  # Remove last channel
    if remove_aux:
        n_channels -= 1
    sample_rate = inlet.info().nominal_srate()
    stream_running = True

    data = np.zeros((N_FFT, n_channels))

    # Create an OSC client
    osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
    shared_list_lock = Lock()

    def compute_fft():
        global fft
        while True:
            t = time.time() + 1 / FFT_COMPUTE_RATE
            if stream_running:
                with shared_list_lock:
                    fft = np.abs(np.fft.rfft(data, axis=0))
                osc_client.send_message("/muse/eeg_fft", list(np.mean(fft, axis=-1)))
            time.sleep(max(0, t - time.time()))

    # Create threads that will compute FFT and send results via OSC
    def get_power_band_to_osc_fn(power_band_name, freq_min, freq_max):
        def power_band_to_osc():
            while True:
                t = time.time() + 1 / FREQUENCY_BANDS_RATE
                if stream_running:
                    with shared_list_lock:
                        data = np.mean(
                            fft[
                                int(freq_min * N_FFT / sample_rate) : int(
                                    freq_max * N_FFT / sample_rate
                                )
                            ]
                        )
                    osc_client.send_message(
                        f"/muse/features/{power_band_name}_absolute", data
                    )
                time.sleep(max(0, t - time.time()))

        return power_band_to_osc

    fft_thread = Thread(target=compute_fft, daemon=True)
    fft_thread.start()

    # ALPHA
    alpha_thread = Thread(
        target=get_power_band_to_osc_fn("alpha", freq_min=8, freq_max=12),
        daemon=True,
        name="alpha",
    )
    alpha_thread.start()

    # BETA
    beta_thread = Thread(
        target=get_power_band_to_osc_fn("beta", freq_min=12, freq_max=30),
        daemon=True,
        name="beta",
    )
    beta_thread.start()

    # Theta
    theta_thread = Thread(
        target=get_power_band_to_osc_fn("theta", freq_min=4, freq_max=8),
        daemon=True,
        name="theta",
    )
    theta_thread.start()

    # Delta
    delta_thread = Thread(
        target=get_power_band_to_osc_fn("delta", freq_min=1, freq_max=4),
        daemon=True,
        name="delta",
    )
    delta_thread.start()

    last_received_time = time.time()
    while True:
        # Read a chunk of samples from LSL
        samples, timestamps = inlet.pull_chunk(
            timeout=LSL_PULL_TIMEOUT, max_samples=LSL_MAX_SAMPLES
        )
        if timestamps:
            last_received_time = time.time()
            stream_running = True
            # Send raw signals
            osc_client.send_message(f"/muse/{stream_type.lower()}", samples[0])

            # Save buffer to compute features
            samples = np.array(samples)[:, ::-1]
            data = np.roll(data, -1, axis=0)
            data[-1, :] = samples
        else:
            if time.time() - last_received_time > 1 / CHECK_STREAM_RATE:
                stream_running = False


def lsl_to_osc(remove_aux):
    streams_types = [stream.type() for stream in pylsl.resolve_streams()]
    if not streams_types:
        raise RuntimeError("Can't find any LSL stream.")

    processes = []

    # Create processes that will stream data from LSL to OSC
    for type in streams_types:
        if type == "EEG":
            process = mp.Process(target=eeg_stream_to_osc, args=(type, remove_aux))
        else:
            process = mp.Process(target=single_lsl_stream_to_osc, args=(type,))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--remove-aux", action="store_true", default=False)
    args, _ = parser.parse_known_args()
    lsl_to_osc(args.remove_aux)
