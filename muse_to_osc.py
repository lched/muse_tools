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


LSL_SCAN_TIMEOUT = 3
LSL_MAX_SAMPLES = 1
LSL_PULL_TIMEOUT = 0.0
OSC_IP = "127.0.0.1"  # Replace with your OSC server IP
OSC_PORT = 9000  # Replace with your OSC server port

# UPDATE RATES (In Hz)
FFT_COMPUTE_RATE = 10
FREQUENCY_BANDS_RATE = 10
CHECK_STREAM_RATE = 10

# FEATURES PARAMETERS
N_FFT = 256
# Should the spectral features be sent as mean values over channels?
SEND_MEAN_FEATURES = True


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


def eeg_stream_to_osc(stream_type, use_aux):
    stream = pylsl.resolve_byprop("type", stream_type, timeout=LSL_SCAN_TIMEOUT)[0]
    # Create LSL inlet
    inlet = pylsl.StreamInlet(stream, max_chunklen=LSL_MAX_SAMPLES)
    n_channels = inlet.info().channel_count()  # Remove last channel
    if not use_aux:
        n_channels -= 1
    sample_rate = inlet.info().nominal_srate()
    stream_running = True

    fft_buffer = np.ones((N_FFT, n_channels)) * 1e-6
    incoming_data = np.zeros(n_channels)

    # Create an OSC client
    osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
    shared_list_lock = Lock()

    def compute_fft():
        global fft
        while True:
            t = time.time() + 1 / FFT_COMPUTE_RATE
            if stream_running:
                with shared_list_lock:
                    fft = np.abs(np.fft.rfft(fft_buffer, axis=0))
                if SEND_MEAN_FEATURES:
                    osc_client.send_message(
                        "/muse/eeg_fft", list(np.mean(fft, axis=-1))
                    )
                else:
                    osc_client.send_message("/muse/eeg_fft", list(fft))
            time.sleep(max(0, t - time.time()))

    # Create threads that will compute FFT and send results via OSC
    def get_power_band_to_osc_fn(power_band_name, freq_min, freq_max):
        def power_band_to_osc():
            idx_min = int(freq_min * N_FFT / sample_rate)
            idx_max = int(freq_max * N_FFT / sample_rate)
            while True:
                t = time.time() + 1 / FREQUENCY_BANDS_RATE

                if not stream_running:
                    continue

                # Get FFT relevant FFT bins from all channels
                with shared_list_lock:
                    power = fft[idx_min:idx_max]
                absolute_power = np.sum(power, axis=0)
                relative_power = absolute_power / np.sum(fft, axis=0)

                if SEND_MEAN_FEATURES:
                    # Compute mean over channels
                    absolute_power = np.mean(absolute_power)
                    relative_power = np.mean(relative_power)

                osc_client.send_message(
                    f"/muse/features/{power_band_name}_absolute", absolute_power
                )
                osc_client.send_message(
                    f"/muse/features/{power_band_name}_relative", relative_power
                )
                time.sleep(max(0, t - time.time()))

        return power_band_to_osc

    fft_thread = Thread(target=compute_fft, daemon=True)
    fft_thread.start()

    # Alpha
    alpha_thread = Thread(
        target=get_power_band_to_osc_fn("alpha", freq_min=8, freq_max=12),
        daemon=True,
        name="alpha",
    )
    alpha_thread.start()

    # Beta
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

    # Gamma
    gamma_thread = Thread(
        target=get_power_band_to_osc_fn("gamma", freq_min=30, freq_max=45),
        daemon=True,
        name="gamma",
    )
    gamma_thread.start()

    last_received_time = time.time()
    while True:
        # Read a chunk of samples from LSL
        samples, timestamps = inlet.pull_chunk(
            timeout=LSL_PULL_TIMEOUT, max_samples=LSL_MAX_SAMPLES
        )

        if timestamps:
            last_received_time = time.time()
            stream_running = True

            if use_aux:
                incoming_data = samples[0]
            else:
                incoming_data = samples[0][:-1]

            # Send raw signals
            osc_client.send_message(f"/muse/{stream_type.lower()}", incoming_data)

            # Save last datat to the fft buffer
            fft_buffer = np.roll(fft_buffer, -1, axis=0)
            fft_buffer[-1, :] = incoming_data
        else:  # timeout mechanism to stop sending data when stream is lost
            if time.time() - last_received_time > 1 / CHECK_STREAM_RATE:
                stream_running = False


def lsl_to_osc(use_aux):
    found_stream = False
    print("Looking for LSL streams...")
    while not found_stream:
        streams_types = [stream.type() for stream in pylsl.resolve_streams()]
        if streams_types:
            found_stream = True

    processes = []

    # Create processes that will stream data from LSL to OSC
    for type in streams_types:
        if type == "EEG":
            process = mp.Process(target=eeg_stream_to_osc, args=(type, use_aux))
        else:
            process = mp.Process(target=single_lsl_stream_to_osc, args=(type,))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--aux", action="store_true", default=False, help="Include Muse AUX channel"
    )
    args, _ = parser.parse_known_args()
    lsl_to_osc(use_aux=args.aux)
