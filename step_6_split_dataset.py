"""
Script chia file JSON thành 4 tập theo tỉ lệ 5-2-2-1
dựa trên độ khó trung bình của từng ảnh.

Input JSON:
{
  "image_id": [["question", "answer", "difficulty"], ...],
  ...
}

Output JSON (mỗi tập):
{
  "image_id": {
    "article_id": "...",
    "article_url": "...",
    "qa": [["question", "answer", "difficulty"], ...]
  },
  ...
}
"""

import json
import re
import argparse
from pathlib import Path

DEFAULT_CAPTION_FILE = Path(__file__).parent / "image_caption_updated.json"
DEFAULT_DATABASE_FILE = Path(__file__).parent.parent / "Eventa/webCrawl/src/merged_7_database.json"


def load_article_mappings(caption_file: Path, database_file: Path) -> dict[str, dict]:
    """Build image_id -> {article_id, article_url} mapping.

    The database file may contain unescaped quotes inside content strings,
    so URLs are extracted via regex instead of full JSON parsing.
    """
    print(f"[0/5] Đang tải mapping article...")

    with open(caption_file, encoding="utf-8") as f:
        captions = json.load(f)
    image_to_article: dict[str, str] = {
        img_id: meta["article_id"]
        for img_id, meta in captions.items()
        if "article_id" in meta
    }
    print(f"      ✓ {len(image_to_article):,} image->article mappings từ caption file")

    print(f"      Đang trích xuất URL từ database (regex)...")
    with open(database_file, encoding="utf-8", errors="replace") as f:
        db_content = f.read()
    url_pattern = re.compile(
        r'"([0-9a-f]{8,})":\s*\{\s*"url":\s*"([^"]+)"',
        re.MULTILINE,
    )
    article_to_url: dict[str, str] = {
        article_id: url for article_id, url in url_pattern.findall(db_content)
    }
    print(f"      ✓ {len(article_to_url):,} article URLs từ database")

    mapping: dict[str, dict] = {}
    for img_id, article_id in image_to_article.items():
        mapping[img_id] = {
            "article_id": article_id,
            "article_url": article_to_url.get(article_id),
        }
    return mapping


def apply_mapping(split_data: dict, mapping: dict[str, dict]) -> dict:
    """Transform split data to final format with article metadata."""
    result = {}
    for img_id, qa_pairs in split_data.items():
        meta = mapping.get(img_id, {"article_id": None, "article_url": None})
        result[img_id] = {
            "article_id": meta["article_id"],
            "article_url": meta["article_url"],
            "qa": qa_pairs,
        }
    return result


def compute_avg_difficulty(qa_pairs: list) -> float:
    """Tính độ khó trung bình cho một ảnh từ danh sách QA pairs."""
    difficulties = []
    for qa in qa_pairs:
        try:
            difficulties.append(float(qa[2]))
        except (IndexError, ValueError, TypeError):
            pass  # Bỏ qua nếu không parse được
    return sum(difficulties) / len(difficulties) if difficulties else 0.0


def split_dataset(input_path: str, output_dir: str, ratio: tuple = (5, 2, 2, 1),
                  split_names: tuple = ("train", "val1", "val2", "test"),
                  seed: int = 42,
                  caption_file: Path = DEFAULT_CAPTION_FILE,
                  database_file: Path = DEFAULT_DATABASE_FILE):
    """
    Chia dataset theo tỉ lệ cho trước.

    Args:
        input_path: Đường dẫn tới file JSON gốc
        output_dir: Thư mục xuất kết quả
        ratio: Tuple tỉ lệ chia (mặc định 5-2-2-1)
        split_names: Tên các tập tương ứng
        seed: Random seed (dùng nếu muốn shuffle trước khi chia)
    """
    article_mapping = load_article_mappings(caption_file, database_file)

    print(f"[1/5] Đang đọc file: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_images = len(data)
    total_qa = sum(len(v) for v in data.values())
    print(f"      ✓ Tổng số ảnh: {total_images:,}")
    print(f"      ✓ Tổng số QA pairs: {total_qa:,}")

    # ── Bước 1: Tính độ khó trung bình cho từng ảnh ──────────────────────────
    print("[2/5] Tính độ khó trung bình cho từng ảnh...")
    image_difficulty = {
        img_id: compute_avg_difficulty(qa_pairs)
        for img_id, qa_pairs in data.items()
    }

    # ── Bước 2: Sắp xếp theo độ khó tăng dần ────────────────────────────────
    print("[3/5] Sắp xếp ảnh theo độ khó trung bình...")
    sorted_ids = sorted(image_difficulty, key=lambda x: image_difficulty[x])

    difficulty_values = list(image_difficulty.values())
    print(f"      ✓ Độ khó min: {min(difficulty_values):.3f}")
    print(f"      ✓ Độ khó max: {max(difficulty_values):.3f}")
    print(f"      ✓ Độ khó trung bình toàn bộ: {sum(difficulty_values)/len(difficulty_values):.3f}")

    # ── Bước 3: Tính số lượng ảnh mỗi tập ───────────────────────────────────
    print("[4/5] Chia dữ liệu theo tỉ lệ", "-".join(map(str, ratio)), "...")
    total_ratio = sum(ratio)
    sizes = []
    remaining = total_images
    for i, r in enumerate(ratio):
        if i == len(ratio) - 1:
            sizes.append(remaining)  # Tập cuối lấy phần còn lại (tránh lệch do làm tròn)
        else:
            n = round(total_images * r / total_ratio)
            sizes.append(n)
            remaining -= n

    splits = []
    start = 0
    for size in sizes:
        splits.append(sorted_ids[start: start + size])
        start += size

    # ── Bước 4: Ghi ra file ──────────────────────────────────────────────────
    print("[5/5] Ghi file kết quả...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_stem = Path(input_path).stem
    stats = []

    for name, ids in zip(split_names, splits):
        split_data = apply_mapping(
            {img_id: data[img_id] for img_id in ids},
            article_mapping,
        )
        out_file = output_path / f"{input_stem}_{name}.json"

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)

        n_qa = sum(len(split_data[i]["qa"]) for i in split_data)
        avg_diff = sum(image_difficulty[i] for i in ids) / len(ids) if ids else 0
        stats.append((name, len(ids), n_qa, avg_diff, out_file))
        print(f"      ✓ [{name}] {len(ids):,} ảnh | {n_qa:,} QA | avg_difficulty={avg_diff:.3f} → {out_file.name}")

    # ── In bảng tổng kết ─────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'Tập':<10} {'Ảnh':>10} {'QA pairs':>12} {'Avg diff':>10} {'%':>6}")
    print("-"*65)
    for name, n_img, n_qa, avg_d, _ in stats:
        pct = n_img / total_images * 100
        print(f"{name:<10} {n_img:>10,} {n_qa:>12,} {avg_d:>10.3f} {pct:>5.1f}%")
    print("="*65)
    print(f"{'TỔNG':<10} {total_images:>10,} {total_qa:>12,}")
    print(f"\nTất cả file đã được lưu vào: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Chia file JSON QA dataset theo độ khó trung bình."
    )
    parser.add_argument(
        "input", help="Đường dẫn file JSON đầu vào", default="./image_vqa_with_difficulty_cleaned.json", nargs="?"
    )
    parser.add_argument(
        "-o", "--output-dir", default="./splits",
        help="Thư mục lưu file output (mặc định: ./splits)"
    )
    parser.add_argument(
        "-r", "--ratio", nargs=4, type=int, default=[5, 2, 2, 1],
        metavar=("R1", "R2", "R3", "R4"),
        help="Tỉ lệ chia (mặc định: 5 2 2 1)"
    )
    parser.add_argument(
        "-n", "--names", nargs=4, default=["train", "val1", "val2", "test"],
        metavar=("N1", "N2", "N3", "N4"),
        help="Tên 4 tập (mặc định: train val1 val2 test)"
    )
    parser.add_argument(
        "--caption-file", default="./image_caption_updated.json",
        help="Đường dẫn file image_caption_updated.json"
    )
    parser.add_argument(
        "--database-file", default="../Eventa/webCrawl/src/merged_7_database.json",
        help="Đường dẫn file merged_7_database.json"
    )
    args = parser.parse_args()

    split_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        ratio=tuple(args.ratio),
        split_names=tuple(args.names),
        caption_file=Path(args.caption_file),
        database_file=Path(args.database_file),
    )


if __name__ == "__main__":
    main()