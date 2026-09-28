import csv
import os
from PIL import Image

PHOTO_DIR = "data/photos"
LABEL_FILE = "data/labels.csv"
TARGET = 50  # how many photos to label

DAMAGE_TYPES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
SEVERITIES = ["minor", "moderate", "severe"]

# Load photos already labeled, so you can stop and continue later
done = set()
if os.path.exists(LABEL_FILE):
    with open(LABEL_FILE, newline="") as f:
        done = {row["photo"] for row in csv.DictReader(f)}

new_file = not os.path.exists(LABEL_FILE)
f = open(LABEL_FILE, "a", newline="")
writer = csv.writer(f)
if new_file:
    writer.writerow(["photo", "part", "damage_type", "severity"])

photos = sorted(p for p in os.listdir(PHOTO_DIR) if p.endswith(".jpg"))

for photo in photos:
    if len(done) >= TARGET:
        break
    if photo in done:
        continue

    Image.open(os.path.join(PHOTO_DIR, photo)).show()  # opens the photo
    print(f"\n[{len(done) + 1}/{TARGET}] {photo}")

    for i, d in enumerate(DAMAGE_TYPES, 1):
        print(f"  {i}. {d}")
    choice = input("Damage type number (s = skip photo, q = quit): ").strip()
    if choice == "q":
        break
    if choice == "s" or not choice.isdigit() or not 1 <= int(choice) <= 6:
        print("  skipped")
        continue
    damage = DAMAGE_TYPES[int(choice) - 1]

    part = input("Car part (e.g. front bumper, rear door, headlight): ").strip()

    for i, s in enumerate(SEVERITIES, 1):
        print(f"  {i}. {s}")
    sev = input("Severity number: ").strip()
    severity = SEVERITIES[int(sev) - 1] if sev in ("1", "2", "3") else "moderate"

    writer.writerow([photo, part, damage, severity])
    f.flush()  # save immediately
    done.add(photo)

f.close()
print(f"\nSaved {len(done)} labels to {LABEL_FILE}")
