VLM_PROMPT = """
Analyze the aerial image.

Identify visible terrain, land cover, structures, and objects.

Rules:

* Report only things that are clearly visible.
* Do not infer, guess, or hallucinate objects.
* Do not describe location, direction, adjacency, boundaries, shape, or spatial relationships.
* Do not describe size.
* Use short semantic descriptions.
* Focus on what is present, not where it is.
* Merge duplicate observations.
* Output valid JSON only.
* Do not include markdown fences, comments, explanations, or extra text.
* Use double quotes for every JSON key and string value.

Return exactly one JSON object and nothing else.

Format:

{
"observations": [
"dense forest",
"grassy field",
"dirt road",
"small building"
]
}
"""

REASONING_PROMPT = """
Task:
{task}

Observations:
{observations}

Existing canonical label vocabulary:
{vocabulary}

Convert the observations into semantic categories suitable for visual grounding and segmentation.

Requirements:

* The generated prompt will be used as a query for a visual grounding model.

* The grounding model converts the prompt into an embedding and matches it against image regions.

* Create prompts that maximize similarity to the target region while remaining visually distinct from competing categories.

* Use the strongest grounding concept for the category, augmented with a small amount of visual detail when it improves discrimination.

* Stay close to common semantic concepts likely to exist in the grounding model's training data.

* Prefer concise descriptive noun phrases.

* Do not use synonym lists.

* Do not use negations.

* Do not describe location, direction, boundaries, adjacency, size, shape, or spatial relationships.

* Avoid decorative, subjective, or overly specific descriptions.

* Reuse labels from the existing vocabulary when possible.

* Merge equivalent concepts.

* Prefer a small number of highly informative categories over many weak categories.

* Favor precision over recall when selecting categories.

* Assign a relevance score from 0-100 representing the likelihood that the task target would be visually located within that region.

* Consider both direct presence and strong environmental associations with the target.

* Output valid JSON only.

* Return exactly one JSON object and nothing else.

* Do not include markdown fences, comments, explanations, or extra text.

* Use double quotes for every JSON key and string value.

* Every top-level value must be an object with exactly "label" and "score".

Format:

{{
"forest canopy": {{
"label": "trees",
"score": 0
}},
"grass field": {{
"label": "field",
"score": 30
}},
"paved roadway": {{
"label": "road",
"score": 90
}},
"building rooftop": {{
"label": "building",
"score": 20
}}
}}
"""
