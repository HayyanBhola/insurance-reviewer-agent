import os
from datasets import load_dataset

os.makedirs("data/photos", exist_ok=True)

# streaming=True downloads images one by one, not the whole 2.2 GB
ds = load_dataset("harpreetsahota/CarDD", split="train", streaming=True)

for i, row in enumerate(ds.take(100)):
    row["image"].convert("RGB").save(f"data/photos/car_{i:03d}.jpg")
    print("saved", i)

print("Done: 100 photos in data/photos/")