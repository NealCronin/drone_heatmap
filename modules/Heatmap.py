from ultralytics.models.sam import SAM3SemanticPredictor
import numpy as np
import cv2

class Heatmap:
    def __init__(self):

        overrides = dict(
            conf=0.25,
            task="segment",
            mode="predict",
            model="models/sam3.pt",
            half=True,  # Use FP16 for faster inference
            save=False,
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)

    def _parse_dict(self, scene_dict):
        labels = list(scene_dict.keys())

        return labels

    def get(self, image, scene_dict):

        labels = self._parse_dict(scene_dict)

        if len(labels) < 1:
            return None

        results = self.predictor(image, text=labels)
        result = results[0]

        # annotated = result.plot()
        # return annotated

        masks = result.masks.data.cpu().numpy()  # (N, H, W)

        heatmap = np.zeros(image.shape[:2], dtype=np.float32)

        for mask, label in zip(masks, labels):
            score = scene_dict[label]

            heatmap += mask.astype(np.float32) * score

        heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
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
