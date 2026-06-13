from pathlib import Path
import cv2
import pandas as pd

from modules.SceneUnderstanding import SceneUnderstanding
from modules.Heatmap import Heatmap

class DroneHeatmap: 
    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        
        self.query_csv = pd.read_csv(self.dataset_root / "query.csv")

        self.query_images_dir = (self.dataset_root / "query_images")

        self.index = 0

        self.scene_understanding = SceneUnderstanding()
        self.heatmap = Heatmap()

    def has_next(self) -> bool:
        return self.index < len(self.query_csv)

    def reset(self):
        self.index = 0

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
    
    def show_video(self, image):

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

            scene_dict = self.scene_understanding.get_labels(image, "Find cars")
            print(scene_dict)

            heatmap = self.heatmap.get(image, scene_dict)
            if heatmap is not None: out = heatmap

            self.show_video(out)
        

if __name__ == "__main__":

    drone = DroneHeatmap(r"C:\Users\jleto\Downloads\Train\Train")

    try:
        while drone.has_next():
            drone.run()

    except Exception as e:
        print(e)

    finally:
        cv2.destroyAllWindows()