import json
import shutil
from pathlib import Path

# ======================
# GLOBAL CONFIG
# ======================

SRC_DIR = "/raid/ltnghia01/phucpv/Eventa/webCrawl/src/database_image"
TARGET_DIR = "/raid/ltnghia01/phucpv/VQA/images"
JSON_FILE = "./database.json"
IMAGE_EXT = ".png"   # đổi nếu cần (.png, .webp...)

# ======================

src_path = Path(SRC_DIR)
target_path = Path(TARGET_DIR)
json_path = Path(JSON_FILE)

if not json_path.exists():
    raise FileNotFoundError(f"JSON file not found: {json_path}")

# Tạo thư mục target nếu chưa có
target_path.mkdir(parents=True, exist_ok=True)

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

copied = 0
missing = 0

for article_id, article in data.items():
    images = article.get("images", [])
    for img in images:
        image_id = img.get("image_id")
        if not image_id:
            continue

        src_file = src_path / f"{image_id}{IMAGE_EXT}"
        target_file = target_path / f"{image_id}{IMAGE_EXT}"

        if src_file.exists():
            shutil.copy2(src_file, target_file)
            copied += 1
            print(f"Copied: {src_file.name}")
        else:
            missing += 1
            print(f"Missing: {src_file.name}")

print("\n=== DONE ===")
print(f"Copied: {copied}")
print(f"Missing: {missing}")