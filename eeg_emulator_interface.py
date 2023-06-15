import multiprocessing as mp
import time
import tkinter as tk
from tkinter import filedialog

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


class WaveformPlayer(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.progress = 0.0  # Progress in percentage (0.0 to 100.0)
        self.progress_interval = 100  # Interval to update progress bar in milliseconds
        self.pool = None
        self.pack()

        self.create_widgets()

    def create_widgets(self):
        self.canvas = tk.Canvas(self, width=800, height=200, bg="white")
        self.canvas.pack()

        self.progress_bar = self.canvas.create_rectangle(0, 0, 0, 0, fill="red")

        self.canvas.bind("<Button-1>", self.load_waveform)

    def load_waveform(self, event):
        xdf_file = filedialog.askopenfilename(filetypes=[("XDF files", "*.xdf")])
        streams, _ = pyxdf.load_xdf(xdf_file)

        if self.pool:
            self.pool.terminate()

        self.pool = mp.Pool(processes=len(streams))

        for stream in streams:
            sample_rate = float(stream["info"]["nominal_srate"][0])
            if stream["info"]["type"][0].lower() == "eeg":
                waveform_data = np.mean(stream["time_series"], axis=-1)
                self.draw_waveform(waveform_data)
                self.duration = len(waveform_data) / sample_rate

            if sample_rate == 0:
                task = launch_events_stream
            else:
                task = launch_sampled_stream
            self.pool.apply_async(task, args=(stream,))
        self.pool.close()
        self.start_progress()

    def draw_waveform(self, waveform_data):
        self.canvas.delete("waveform")

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        x = np.linspace(0, width, len(waveform_data))
        # Scale the waveform to fit the canvas
        y = waveform_data / np.max(np.abs(waveform_data)) * height / 2 + height / 2
        points = np.column_stack((x, y)).flatten().tolist()
        self.canvas.create_line(points, fill="blue", tags="waveform")

    def start_progress(self):
        self.progress = 0.0
        self.update_progress()

    def update_progress(self):
        if self.progress < 100.0:
            self.progress += self.progress_interval / (
                self.duration * 10
            )  # Increase progress by interval
            self.canvas.coords(
                self.progress_bar,
                0,
                0,
                self.canvas.winfo_width() * (self.progress / 100),
                self.canvas.winfo_height(),
            )
            self.after(self.progress_interval, self.update_progress)


if __name__ == "__main__":
    root = tk.Tk()
    app = WaveformPlayer(root)
    app.mainloop()
