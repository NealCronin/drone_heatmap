import os

os.environ["HF_HOME"] = r"D:\huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\huggingface\hub"
os.environ["TRANSFORMERS_CACHE"] = r"D:\huggingface\transformers"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_JIT"] = "0"

from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import time

MODEL_ID = "allenai/Molmo2-4B"
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available in this Python environment.")

DEVICE = torch.device("cuda:0")
DTYPE = torch.float16
START_TIME = time.perf_counter()


def log_step(message):
    elapsed = time.perf_counter() - START_TIME
    print(f"[{elapsed:7.2f}s] {message}", flush=True)


def print_cuda_memory(label):
    free_bytes, total_bytes = torch.cuda.mem_get_info(DEVICE)
    allocated_bytes = torch.cuda.memory_allocated(DEVICE)
    reserved_bytes = torch.cuda.memory_reserved(DEVICE)
    gib = 1024**3

    print(
        f"{label}: free={free_bytes / gib:.2f} GiB, "
        f"total={total_bytes / gib:.2f} GiB, "
        f"allocated={allocated_bytes / gib:.2f} GiB, "
        f"reserved={reserved_bytes / gib:.2f} GiB"
    )


log_step(f"CUDA device: {torch.cuda.get_device_name(DEVICE)}")
print_cuda_memory("Before load")

log_step("Loading processor...")
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

log_step("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    dtype=DTYPE,
    device_map={"": DEVICE},
    low_cpu_mem_usage=True,
)
model.eval()
log_step(f"Loaded model on {DEVICE} with dtype={DTYPE}.")
print_cuda_memory("After load")

first_param = next(model.parameters())
log_step(f"First parameter device={first_param.device}, dtype={first_param.dtype}")

image = Image.open("001024.png").convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": "Point to the roads in this aerial image."
            },
        ],
    }
]

log_step("Preparing inputs...")

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)

inputs = {
    k: v.to(DEVICE, non_blocking=True)
    if hasattr(v, "to")
    else v
    for k, v in inputs.items()
}
print_cuda_memory("After inputs")

log_step("Generating...")

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )
torch.cuda.synchronize(DEVICE)
log_step("Finished generation.")
print_cuda_memory("After generation")

generated_tokens = output[0, inputs["input_ids"].size(1):]

generated_text = processor.tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True,
)

print("\n=== RESPONSE ===")
print(generated_text)
