from pathlib import Path
import cv2
import pandas as pd
import traceback

from modules.SceneUnderstanding import SceneUnderstanding
from modules.Heatmap import Heatmap
from scripts.video_output import create_video_writer, release_video_writer

class DroneHeatmap: 
    def __init__(self, dataset_root: str, sam_step=15):
        self.dataset_root = Path(dataset_root)
        self.sam_step = sam_step
        
        self.query_csv = pd.read_csv(self.dataset_root / "query.csv")

        self.query_images_dir = (self.dataset_root / "query_images")

        self.index = 0

        self.scene_understanding = SceneUnderstanding()
        self.heatmap = Heatmap()
        self.video_writer = None
        self.video_path = None

    def has_next(self) -> bool:
        return self.index < len(self.query_csv)

    def reset(self):
        self.index = 0

    def should_run_sam(self, frame):
        return frame["frame_index"] % self.sam_step == 0

    def get_next_frame(self):
        if not self.has_next():
            return None

        row = self.query_csv.iloc[self.index]

        image_path = (self.query_images_dir / row["name"])

        image = cv2.imread(str(image_path))
        image[:, :, 1] = (image[:, :, 1] * 0.65).astype(image.dtype)
        image[:, :, 2:3] = (image[:, :, 2:3] * 0.8).astype(image.dtype)

        frame = {
            "image": image,
            "image_path": str(image_path),
            "easting": row["easting"],
            "northing": row["northing"],
            "altitude": row["altitude"],
            "orientation": (
                row["orient_x"],
                row["orient_y"],
                row["orient_z"],
                row["orient_w"],
            ),
            "frame_index": self.index,
        }

        self.index += 1

        return frame
    
    def _get_video_writer(self, image):
        if self.video_writer is not None:
            return self.video_writer

        self.video_writer, self.video_path = create_video_writer(image)
        return self.video_writer

    def close_video(self):
        release_video_writer(self.video_writer)
        self.video_writer = None

    def show_video(self, image):

        self._get_video_writer(image).write(image)

        cv2.imshow("Frame", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    def run(self):

        if self.has_next():

            frame = self.get_next_frame()

            image = frame["image"]
            out = image

            gps = (
                frame["easting"],
                frame["northing"]
            )
            # print(
            #     frame["frame_index"],
            #     gps,
            #     frame["altitude"]
            # )

            scene_dict = None
            if self.should_run_sam(frame):
                scene_dict = self.scene_understanding.get_labels(image, "Find cars")
            # print(scene_dict)

            heatmap = self.heatmap.get(image, scene_dict)
            if heatmap is not None: out = heatmap

            self.show_video(out)
        

if __name__ == "__main__":

    drone = DroneHeatmap(r"C:\Users\jleto\Downloads\Train\Train")

    try:
        while drone.has_next():
            drone.run()

    except Exception:
        traceback.print_exc()

    finally:
        drone.close_video()
        cv2.destroyAllWindows()
