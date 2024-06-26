from PIL import Image, ImageSequence
import numpy as np
import einops
import numpy as np
import cv2
from pathlib import Path


def read_gif(filepath, mode="RGB"):
    """
    Reads a GIF file and returns a list of NumPy arrays representing each frame.

    Args:
        filepath: Path to the GIF file.
        mode: Mode of the image (default: 'RGBA').

    Returns:
        A list of NumPy arrays, each representing a frame of the GIF.
    """
    with Image.open(filepath) as im:
        frames = np.stack([np.array(frame.convert(mode)) for frame in ImageSequence.Iterator(im)])
    return frames


def save_numpy_array_as_video(filepath, frames, fps=24):
    """
    Saves a NumPy array of frames as a video using OpenCV.

    Args:
        filepath: Path to save the video file.
        frames: NumPy array of shape (T, H, W, C) representing video frames.
        fps: Frames per second (default: 24).
    """
    # Get frame size from the first frame
    height, width, channels = frames[0].shape[:3]

    # Define video writer with fourcc code (e.g., 'mp4v' for MP4)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Change codec as needed (e.g., 'XVID')
    out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

    # Write each frame to the video
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Ensure frame data type is uint8

    # Release video writer
    out.release()


def process_finite(folder_data):
    for k, v in folder_data.items():
        video = np.stack([read_gif(gif_path) for gif_path in v])
        b, t, h, w, _ = video.shape
        w = w // 2
        video = video[:, :, :, :w, :]
        dup_t = 144 // t
        dup_h = 128 // h
        dup_w = 128 // w
        video = einops.repeat(
            video, "b t h w c -> b (t dup_t) (h dup_h) (w dup_w) c", dup_t=dup_t, dup_h=dup_h, dup_w=dup_w
        )
        edge_pixel = 2
        video = np.pad(video, ((0, 0), (0, 0), (edge_pixel, edge_pixel), (edge_pixel, edge_pixel), (0, 0)))
        b, t, h, w, _ = video.shape
        video = einops.rearrange(video, "(gr gc) t h w c -> t (gr h) (gc w) c", gr=2, gc=8)
        print(k, video.shape)
        save_numpy_array_as_video(f"{root_path}/{k}.mp4", video, fps=12)


def process_infinite(folder_data):
    for k, v in folder_data.items():
        video = np.stack([read_gif(gif_path) for gif_path in v])
        b, t, h, w, _ = video.shape
        w = w // 2
        video = video[:, :, :, :w, :]
        dup_t = 1
        dup_h = 256 // h
        dup_w = 256 // w
        video = einops.repeat(
            video, "b t h w c -> b (t dup_t) (h dup_h) (w dup_w) c", dup_t=dup_t, dup_h=dup_h, dup_w=dup_w
        )
        edge_pixel = 2
        video = np.pad(video, ((0, 0), (0, 0), (edge_pixel, edge_pixel), (edge_pixel, edge_pixel), (0, 0)))
        b, t, h, w, _ = video.shape
        video = einops.rearrange(video, "(gr gc) t h w c -> t (gr h) (gc w) c", gr=b // 4, gc=4)
        print(k, video.shape)
        save_numpy_array_as_video(f"{root_path}/{k}.mp4", video, fps=12)


def get_folder_dict(root_path):
    folder_data = {}
    for item in Path(root_path).iterdir():
        if item.is_dir():
            # Get filenames within the subdirectory
            filenames = [f for f in item.iterdir() if f.is_file()]
            filenames = sorted(filenames, key=lambda x: int(x.name.split("_")[1]))
            folder_data[item.name] = filenames
    return folder_data


# root_path = "/Users/boyuan/Documents/research/diffusion_forcing/website_videos/video_finite"
root_path = "/Users/boyuan/Documents/research/diffusion_forcing/website_videos/video_infinite"
folder_data = get_folder_dict(root_path)

# process_finite(folder_data)
process_infinite(folder_data)
