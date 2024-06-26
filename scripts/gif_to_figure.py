from PIL import Image, ImageSequence
import numpy as np
import matplotlib.pyplot as plt
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


def read_video_to_frames(video_path):
    """
    Reads an mp4 video and returns a NumPy array of frames.

    Args:
        video_path: Path to the video file.

    Returns:
        A NumPy array of shape (num_frames, height, width, channels), where
        each element is a frame from the video.
    """

    # Open the video capture object
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
        return None

    # Store frames in a list
    frames = []
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If the frame is read correctly retrieve channels
        if ret:
            # BGR to RGB conversion (optional)
            # if frame.shape[0] > frame.shape[1]:
            #     frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

            # Exit if press 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

    # Convert the frame list into a NumPy array
    frames_np = np.array(frames)

    return frames_np


def process_video_prediction_finite(folder_data):
    for k, v in folder_data.items():
        if "df" not in k:
            continue
        video = np.stack([read_gif(gif_path) for gif_path in v])
        b, t, h, w, _ = video.shape
        w = w // 2
        video = video[:, :, :, :w, :]
        dup_t = 1
        dup_h = 1
        dup_w = 1
        subsample = 4 if "minecraft" in k else 1
        video = einops.repeat(
            video, "b t h w c -> b (t dup_t) (h dup_h) (w dup_w) c", dup_t=dup_t, dup_h=dup_h, dup_w=dup_w
        )
        video = video[:, ::subsample]
        edge_pixel = 2
        video = np.pad(video, ((0, 0), (0, 0), (edge_pixel, edge_pixel), (edge_pixel, edge_pixel), (0, 0)))
        b, t, h, w, _ = video.shape
        video = einops.rearrange(video, "b (gr gc) h w c -> b (gr h) (gc w) c", gr=t // 9, gc=9)
        for i in range(2):
            cv2.imwrite(f"{root_path}/{k}_{i}.png", cv2.cvtColor(video[i], cv2.COLOR_RGB2BGR))


def process_video_prediction_infinite(folder_data):
    for k, v in folder_data.items():
        if "df" not in k:
            continue
        video = np.stack([read_gif(gif_path) for gif_path in v])
        b, t, h, w, _ = video.shape
        w = w // 2
        video = video[:, :, :, :w, :]
        dup_t = 1
        dup_h = 128 // h
        dup_w = 128 // w
        subsample = 4 if "minecraft" in k else 2
        video = einops.repeat(
            video, "b t h w c -> b (t dup_t) (h dup_h) (w dup_w) c", dup_t=dup_t, dup_h=dup_h, dup_w=dup_w
        )
        video = video[:, :180:subsample]
        edge_pixel = 2
        video = np.pad(video, ((0, 0), (0, 0), (edge_pixel, edge_pixel), (edge_pixel, edge_pixel), (0, 0)))
        b, t, h, w, _ = video.shape
        video = einops.rearrange(video, "b (gr gc) h w c -> b (gr h) (gc w) c", gr=t // 9, gc=9)
        for i in [0, 3]:
            cv2.imwrite(f"{root_path}/{k}_{i}.png", cv2.cvtColor(video[i], cv2.COLOR_RGB2BGR))


def process_planning_video(folder_data):
    for k, v in folder_data.items():
        for video_path in v:
            if "df" not in video_path.name or ".mp4" not in video_path.suffix:
                continue
            video = read_video_to_frames(str(video_path.absolute()))
            t, h, w, _ = video.shape
            dup_t = 1
            dup_h = 1
            dup_w = 1
            video = einops.repeat(
                video, "t h w c -> (t dup_t) (h dup_h) (w dup_w) c", dup_t=dup_t, dup_h=dup_h, dup_w=dup_w
            )
            if k == "convergence":
                video = video[-64::8]
            else:
                video = video[::32]
            t, h, w, _ = video.shape
            print(video.shape)
            gc = int(input("Numer of cols:"))
            video = einops.rearrange(video, "(gr gc) h w c -> (gr h) (gc w) c", gr=t // gc, gc=gc)
            print(video_path)
            cv2.imwrite(str(video_path.with_suffix(".png").absolute()), cv2.cvtColor(video, cv2.COLOR_RGB2BGR))


def get_folder_dict(root_path, sort=True):
    folder_data = {}
    for item in Path(root_path).iterdir():
        if item.is_dir():
            # Get filenames within the subdirectory
            filenames = [f for f in item.iterdir() if f.is_file()]
            if sort:
                filenames = sorted(filenames, key=lambda x: int(x.name.split("_")[1]))
            folder_data[item.name] = filenames
    return folder_data


# process_finite(folder_data)

# root_path = "/Users/boyuan/Documents/research/diffusion_forcing/website_videos/video_finite"
# folder_data = get_folder_dict(root_path)
# process_video_prediction_finite(folder_data)
# root_path = "/Users/boyuan/Documents/research/diffusion_forcing/website_videos/video_infinite"
# folder_data = get_folder_dict(root_path)
# process_video_prediction_infinite(folder_data)
root_path = "/Users/boyuan/Documents/research/diffusion_forcing/website_videos/planning"
folder_data = get_folder_dict(root_path, sort=False)
process_planning_video(folder_data)
