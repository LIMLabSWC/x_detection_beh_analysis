import argparse
import platform
import tkinter as tk
from tkinter import filedialog, messagebox
import skvideo.io
import numpy as np
from PIL import Image, ImageTk
import csv
import pandas as pd
from pathlib import Path
import yaml
from mousepipeline import posix_from_win

class VideoFrameExtractor:
    def __init__(self, root,vids=None):
        self.root = root
        self.root.title("Video Frame Extractor")

        self.select_button = tk.Button(self.root, text="Select Video", command=self.load_video)
        self.select_button.pack(pady=20)

        self.save_button = tk.Button(self.root, text="Save Points", command=self.save_points)
        self.save_button.pack(pady=20)

        self.next_video_button = tk.Button(self.root, text="Load Next Video", command=self.load_next_video)
        self.next_video_button.pack(pady=20)

        self.video_path = None
        self.frames = []
        self.current_frame_index = 0
        self.points = []
        self.all_points = []

        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack()

        self.next_button = tk.Button(self.root, text="Next Frame", command=self.show_next_frame)
        self.next_button.pack(pady=20)

        self.canvas.bind("<Button-1>", self.mark_point)
        self.root.bind("<Right>", self.show_next_frame_key)
        self.root.bind("<Left>", self.show_previous_frame_key)

        self.vids = vids
        self.current_vid_i = 0

    def load_video(self):
        if self.vids:
            self.video_path = filedialog.askopenfilename(initialdir=self.vids[self.current_vid_i],
                                                         filetypes=[("Video files", "*.mp4;*.csv")])

        else:
            self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.csv")])
        if self.video_path:
            csv_path = Path(self.video_path).parent/ "pupil_points.csv"
            if csv_path.is_file():
                choice = messagebox.askquestion("CSV Exists",
                                                "A CSV file already exists for this video. Do you want to redo marking on this video?")
                if choice == "yes":
                    self.load_existing_csv(csv_path)
                else:
                    self.load_next_video()
            else:
                self.extract_frames()
                self.show_frame()
        else:
            messagebox.showerror("Error", "No video selected")

    def load_existing_csv(self, csv_path):
        self.frames = []
        self.all_points = []

        with open(csv_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                frame_index = int(row[0])
                if frame_index >= len(self.frames):
                    self.frames.extend([None] * (frame_index - len(self.frames) + 1))
                    self.all_points.extend([[]] * (frame_index - len(self.all_points) + 1))
                points = [(int(row[i]), int(row[i + 1])) for i in range(1, len(row), 2)]
                self.all_points[frame_index] = points

        self.current_frame_index = 0
        self.show_frame()

    def load_next_video(self):
        past_i = self.vids.index(Path(self.video_path).parent)
        self.video_path = None
        self.frames = []
        self.current_frame_index = 0
        self.points = []
        self.all_points = []
        self.canvas.delete("all")
        if self.vids:
            self.current_vid_i = past_i + 1
        self.load_video()

    def extract_frames(self):
        video_reader = skvideo.io.vreader(self.video_path, num_frames=10000)
        all_frames = np.array([frame for frame in video_reader])
        total_frames = len(all_frames)
        frame_indices = np.random.choice(total_frames, 50, replace=False)
        self.frame_indices = frame_indices
        self.frames = [all_frames[idx] for idx in frame_indices]
        self.current_frame_index = 0
        self.all_points = [[] for _ in range(len(self.frames))]  # Initialize all_points to store points for each frame

    def show_frame(self):
        if 0 <= self.current_frame_index < len(self.frames):
            frame = self.frames[self.current_frame_index]
            frame_image = Image.fromarray(frame)
            frame_image = frame_image.resize((800, 600), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(frame_image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            # Redraw existing points
            self.points = self.all_points[self.current_frame_index]
            for i, (x, y) in enumerate(self.points):
                self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="red")
                self.canvas.create_text(x, y-10, text=f"{i+1}", fill="red")

    def show_next_frame(self):
        if self.current_frame_index < len(self.frames) - 1:
            self.current_frame_index += 1
            self.show_frame()
        else:
            messagebox.showinfo("Info", "No more frames")

    def show_next_frame_key(self, event):
        self.show_next_frame()

    def show_previous_frame(self):
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.show_frame()
        else:
            messagebox.showinfo("Info", "This is the first frame")

    def show_previous_frame_key(self, event):
        self.show_previous_frame()

    def mark_point(self, event):
        if len(self.points) < 8:
            x, y = event.x, event.y
            self.points.append((x, y))
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="red")
            self.canvas.create_text(x, y-10, text=f"{len(self.points)}", fill="red")
            self.all_points[self.current_frame_index] = self.points  # Save points for the current frame
            print(f"Point {len(self.points)}: ({x}, {y})")
        if len(self.points) == 8:
            print("Points for current frame:", self.points)

    def save_points(self):
        if not self.video_path:
            messagebox.showerror("Error", "No video selected")
            return

        # if any(len(points) != 4 for points in self.all_points):
        #     messagebox.showerror("Error", "Not all frames have 4 points marked")
        #     return

        video_dir = Path(self.video_path).parent
        csv_path = video_dir/ "pupil_points.csv"

        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Frame"]+sum([[f'x{i}', f'y{i}'] for i in range(1, 9)],[]))
            for i, points in enumerate(self.all_points):
                writer.writerow(self.frame_indices[i] + [coord for point in points for coord in point])

        messagebox.showinfo("Info", f"Points saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default=Path('config', 'mouse_fam_old_conf_unix.yaml'))

    args = parser.parse_args()
    sys_os = platform.system().lower()

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    ceph_dir = Path(config[f'ceph_dir_{sys_os}'])
    if config.get('session_topology_path'):
        use_session_topology = True
    else:
        use_session_topology = False

    if use_session_topology:
        sess_top_path = ceph_dir / posix_from_win(config['session_topology_path'])
        session_topology = pd.read_csv(sess_top_path)
        animals2process = session_topology['name'].unique().tolist()
        dates2process = session_topology['date'].unique().astype(str).tolist()
    else:
        session_topology = None
    start_date = 0
    vids2process = (session_topology.query('date >= @start_date').sort_values(['date','name'])['videos_dir'].tolist())
    vids2process = [ceph_dir/posix_from_win(vid) for vid in vids2process if
                    all(['pupil_bbox.csv' not in e.name for e in  list((ceph_dir/posix_from_win(vid)).iterdir())])]
    print(f'first video: {vids2process[0]}')

    root = tk.Tk()
    app = VideoFrameExtractor(root, vids=vids2process)
    root.mainloop()
