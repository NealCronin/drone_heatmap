from pathlib import Path
import time

import cv2


def timestamped_video_path(output_dir="example"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    return output_dir / f"{timestamp}.mp4"


def create_video_writer(image, output_dir="example", fps=30):
    video_path = timestamped_video_path(output_dir)

    height, width = image.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    if not video_writer.isOpened():
        video_writer.release()
        raise RuntimeError(f"Failed to open video writer for {video_path}")

    return video_writer, video_path


def release_video_writer(video_writer):
    if video_writer is not None:
        video_writer.release()
