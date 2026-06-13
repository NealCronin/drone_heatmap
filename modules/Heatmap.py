from ultralytics.models.sam import SAM3SemanticPredictor
import numpy as np
import cv2
from dataclasses import dataclass

@dataclass
class Region:
    mask: np.ndarray
    label: str
    score: float

class Heatmap:
    def __init__(self, sam_step=15):

        self.sam_step = sam_step
        self.frame_idx = 0

        overrides = dict(
            conf=0.5,
            task="segment",
            mode="predict",
            model="models/sam3.pt",
            half=True,  # Use FP16 for faster inference
            save=False,
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)

        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.prev_gray = None

    def _parse_dict(self, scene_dict):
        labels = list(scene_dict.keys())

        return labels
    
    def _get_flow_map(self, curr_image):
        
        curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)

        flow = self.dis.calc(self.prev_gray, curr_gray, None)

        h, w = curr_image.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        map_x = (x - flow[..., 0]).astype(np.float32)
        map_y = (y - flow[..., 1]).astype(np.float32)

        self.prev_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)

        return map_x, map_y
    
    def _create_region(self, mask: np.ndarray, label, score):
        self.regions.append(
            Region(
                mask=mask,
                label=label,
                score=score
            )
        )

    def _create_heatmap(self, image):
        heatmap = np.zeros(image.shape[:2], dtype=np.float32)
        valid = np.zeros(image.shape[:2], dtype=np.float32)

        for region in self.regions:
            mask = region.mask.astype(np.float32)
            score = region.score

            heatmap = np.maximum(
                heatmap,
                mask * score
            )

            valid = np.maximum(
                valid,
                mask
            )

        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        valid = cv2.GaussianBlur(valid, (15, 15), 0)
        heatmap = heatmap / (valid + 1e-6)

        heatmap = np.clip(heatmap, 0, 100)
        heatmap = (heatmap * 2.55).astype(np.uint8)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        output = cv2.addWeighted(
            image,    # base image
            0.6,      # weight of base image
            heatmap,  # heatmap overlay image
            0.4,      # weight of heatmap
            0         # constant brightness offset added to every pixel
        )

        return output

    def get(self, image, scene_dict):

        if self.frame_idx % self.sam_step != 0: # Not sam step

            if self.prev_gray is None: return None

            map_x, map_y = self._get_flow_map(image)

            for region in self.regions:
                region.mask = cv2.remap(
                    region.mask.astype(np.uint8),  # mask being tracked
                    map_x,                         # x-coordinate lookup table from optical flow
                    map_y,                         # y-coordinate lookup table from optical flow
                    interpolation=cv2.INTER_NEAREST,  # preserve binary mask values (0/1)
                    borderMode=cv2.BORDER_CONSTANT,   # pixels outside image become a constant value
                    borderValue=0                    # outside-image pixels become background
                )

        else:
            self.prev_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            labels = self._parse_dict(scene_dict)

            if len(labels) < 1:
                return None
        
            results = self.predictor(image, text=labels)
            if not results: return None
            result = results[0]

            if result.masks is None: return None
            masks = result.masks.data.cpu().numpy()  # (N, H, W)

            self.regions = []
            for i in range(len(result.boxes)):
                label = result.names[int(result.boxes.cls[i])]
                mask = masks[i]
                score = scene_dict[label]

                self._create_region(mask, label, score)

            
                # annotated = result.plot()
                # return annotated

        self.frame_idx += 1

        heatmap = self._create_heatmap(image)

        return heatmap
