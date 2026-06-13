from openai import OpenAI
import cv2
import numpy as np
import base64
import json

import os

class SceneUnderstanding:
    def __init__(self, model="gpt-4.1-mini"):
        self.model = model
        self.client = OpenAI()

    def get_labels(self, image: np.ndarray, task: str):

        # debug
        return {
            "forest" : 10,
            "grassland": 70,
            "house" : 5, 
            "road" : 80,
            "river" : 1
        }

        # image = cv2.resize(
        #     image,
        #     (384, 384),
        #     interpolation=cv2.INTER_AREA
        # )

        _, buffer = cv2.imencode(".jpg", image)

        image_b64 = base64.b64encode(
            buffer
        ).decode("utf-8")

        prompt = f"""
            Task: {task}

            Analyze the aerial image.

            Identify only visible regions that have a physical location and extent in the image.

            Rules:
            - Do not detect the task target.
            - Do not include the task target as a label.
            - Only include regions that are visibly present.
            - Every label must correspond to a region that could be spatially localized in the image.
            - Do not output scene descriptions, attributes, qualities, or abstractions.
            - Keys must be a single lowercase word.
            - Values must be relevance scores from 0-100 for accomplishing the task.

            Output requirements:
            - Return exactly one JSON object.
            - Return only JSON.
            - Do not use markdown.
            - Do not use code fences.
            - Do not provide explanations.
            - Do not provide multiple answers.
            - The first character of the response must be '{{'.
            - The last character of the response must be '}}'.
        """

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    ],
                }
            ],
        )

        data = json.loads(response.output_text)

        print(data)

        return data