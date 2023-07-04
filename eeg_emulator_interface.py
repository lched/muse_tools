import multiprocessing as mp
import time
import tkinter as tk
from argparse import ArgumentParser
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
    def __init__(self, master, show_markers=False):
        super().__init__(master)
        self.master = master
        self.show_markers = show_markers
        self.progress = 0.0  # Progress in percentage (0.0 to 100.0)
        self.progress_interval = 100  # Interval to update progress bar in milliseconds
        self.loop_var = tk.BooleanVar()
        self.filename = None
        self.pool = None
        self.pack()

        self.create_widgets()

    def create_widgets(self):
        self.canvas = tk.Canvas(self, width=800, height=200, bg="white")
        self.canvas.pack()
        self.progress_bar = self.canvas.create_rectangle(0, 0, 0, 0, fill="#FFCF67")
        self.canvas.bind("<Button-1>", self.load_waveform)

        # Duration label
        self.duration_label = tk.Label(self, text="Duration: 0.0 seconds")
        self.duration_label.pack(side=tk.LEFT)

        self.loop_checkbox = tk.Checkbutton(
            self,
            text="Loop",
            variable=self.loop_var,  # command=self.toggle_loop
        )
        self.loop_checkbox.pack(side=tk.RIGHT)

    def stop_loop(self):
        if self.pool:
            self.pool.terminate()
            self.pool = None

    def load_waveform(self, event):
        if not self.loop_var.get():
            self.stop_loop()

        if not self.filename or event:
            self.filename = filedialog.askopenfilename(
                filetypes=[("XDF files", "*.xdf")]
            )
            is_first_loop = True
        else:
            is_first_loop = False

        if self.filename:
            streams, _ = pyxdf.load_xdf(self.filename)

            # If there are still processes running, terminate them
            if self.pool:
                self.pool.terminate()
            self.pool = mp.Pool(processes=len(streams))

            markers = None
            for stream in streams:
                sample_rate = float(stream["info"]["nominal_srate"][0])
                if stream["info"]["type"][0].lower() == "eeg":
                    waveform_data = np.mean(stream["time_series"], axis=-1)
                    self.duration = len(waveform_data) / sample_rate

                if stream["info"]["type"][0].lower() == "markers" and self.show_markers:
                    markers = zip(
                        stream["time_stamps"] - stream["time_stamps"][0],
                        stream["time_series"],
                    )

                if sample_rate == 0:
                    task = launch_events_stream
                else:
                    task = launch_sampled_stream

                self.pool.apply_async(task, args=(stream,), callback=self.load_waveform)
            self.duration = len(waveform_data) / sample_rate
            self.duration_label.config(text=f"Duration: {self.duration:.1f} seconds")
            self.draw_waveform(waveform_data, markers)
            if is_first_loop or self.loop_var.get():
                self.start_progress()
            else:
                self.stop_loop()

    def draw_waveform(self, waveform_data, markers=None):
        self.canvas.delete("waveform")

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        x = np.linspace(0, width, len(waveform_data))
        # Scale the waveform to fit the canvas
        y = waveform_data / np.max(np.abs(waveform_data)) * height / 2 + height / 2
        points = np.column_stack((x, y)).flatten().tolist()
        self.canvas.create_line(points, fill="#91D8F0", tags="waveform")

        if markers:
            for marker in markers:
                timestamp, label = marker
                x_position = int(timestamp / self.duration * width)
                self.canvas.create_line(
                    x_position,
                    0,
                    x_position,
                    height,
                    fill="#68293C",
                    width=1,
                    tags="waveform",
                )
                self.canvas.create_text(
                    x_position + 10,
                    height - 20,
                    anchor="n",
                    text=label,
                    fill="#4F315D",
                    font=("Arial", 8),
                    tags="waveform",
                )

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
    parser = ArgumentParser()
    parser.add_argument("-M", "--markers", action="store_true", help="Show markers")
    args, _ = parser.parse_known_args()
    root = tk.Tk()
    photo = tk.PhotoImage(file="./assets/artboard.png")
    root.wm_iconphoto(False, photo)
    app = WaveformPlayer(root, show_markers=args.markers)
    app.mainloop()
