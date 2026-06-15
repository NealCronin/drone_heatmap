import cv2
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch

model_id = "microsoft/Florence-2-large"


def get_detection_payload(parsed, task_prompt):
    if isinstance(parsed, dict) and task_prompt in parsed:
        return parsed[task_prompt]
    return parsed


def get_boxes_and_labels(parsed, task_prompt):
    payload = get_detection_payload(parsed, task_prompt)
    if not isinstance(payload, dict):
        return [], []

    boxes = payload.get("bboxes") or payload.get("boxes") or []
    labels = (
        payload.get("bboxes_labels")
        or payload.get("labels")
        or payload.get("classes")
        or []
    )

    if not labels:
        labels = [text_prompt] * len(boxes)

    return boxes, labels


def get_polygons_and_labels(parsed, task_prompt):
    payload = get_detection_payload(parsed, task_prompt)
    if not isinstance(payload, dict):
        return [], []

    polygons = payload.get("polygons") or payload.get("masks") or []
    labels = (
        payload.get("polygons_labels")
        or payload.get("labels")
        or payload.get("classes")
        or []
    )

    if not labels:
        labels = [text_prompt] * len(polygons)

    return polygons, labels


def normalize_polygon(polygon):
    if not polygon:
        return []

    if isinstance(polygon[0], (int, float)):
        return [
            [polygon[index], polygon[index + 1]]
            for index in range(0, len(polygon) - 1, 2)
        ]

    return polygon


def draw_detections(image, parsed, task_prompt, image_path):
    boxes, labels = get_boxes_and_labels(parsed, task_prompt)
    polygons, polygon_labels = get_polygons_and_labels(parsed, task_prompt)

    canvas = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    overlay = canvas.copy()

    polygon_count = 0
    for index, polygon_group in enumerate(polygons):
        if not polygon_group:
            continue

        if polygon_group and isinstance(polygon_group[0], (int, float)):
            polygon_group = [polygon_group]

        label = polygon_labels[index] if index < len(polygon_labels) else text_prompt

        for polygon in polygon_group:
            points = normalize_polygon(polygon)
            if len(points) < 3:
                continue

            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], (0, 180, 255))
            cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

            x, y = pts.reshape((-1, 2)).min(axis=0)
            cv2.putText(
                canvas,
                f"{polygon_count}: {label}",
                (int(x), max(15, int(y) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            polygon_count += 1

    if polygon_count:
        canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)

    for index, box in enumerate(boxes):
        if len(box) < 4:
            continue

        x1, y1, x2, y2 = [int(round(float(value))) for value in box[:4]]
        label = labels[index] if index < len(labels) else text_prompt
        label = f"{index}: {label}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )
        y_text = max(y1, text_height + baseline + 4)
        cv2.rectangle(
            canvas,
            (x1, y_text - text_height - baseline - 4),
            (x1 + text_width + 6, y_text + baseline),
            (0, 255, 255),
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 3, y_text - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imshow(f"Florence detections - {image_path}", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return len(boxes), polygon_count

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map="auto",
    attn_implementation="eager",
)

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True
)
model.eval()

image_path = "001024.png"
image = Image.open(image_path).convert("RGB")
print(f"Image: {image_path} ({image.size[0]}x{image.size[1]}, {image.mode})")

task_prompt = "<REFERRING_EXPRESSION_SEGMENTATION>"
text_prompt = "roads"
prompt = task_prompt + text_prompt

inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt"
)
inputs = {
    k: v.to(model.device) if hasattr(v, "to") else v
    for k, v in inputs.items()
}

with torch.inference_mode():
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=1,
        use_cache=False,
    )

generated_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=False
)[0]

print(generated_text)

parsed = processor.post_process_generation(
    generated_text,
    task=task_prompt,
    image_size=image.size,
)

print(parsed)

box_count, polygon_count = draw_detections(image, parsed, task_prompt, image_path)
print(f"Visualized {box_count} box(es) and {polygon_count} polygon(s).")
