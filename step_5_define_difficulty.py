import os
import json
import argparse
import torch
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from internvl import CustomQwenVLCaptionModel
import time


def render_difficulty_prompt(template_dir, article_content, qa_pairs):
    env = Environment(loader=FileSystemLoader(template_dir), trim_blocks=True, lstrip_blocks=True)
    template = env.get_template("difficulty_rating.j2")
    return template.render(content=article_content[:5000], qa_pairs=qa_pairs)


def save_json(data, path):
    temp_path = path + ".tmp"
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(temp_path, path)


def find_image_path(image_dir, image_id):
    for ext in ['.png', '.jpg', '.jpeg', '.JPG', '.PNG', '.webp']:
        path = os.path.join(image_dir, f"{image_id}{ext}")
        if os.path.exists(path):
            return path
    return None


def reconcile_with_cleaned_vqa(output_data, vqa_data, verbose=True):
    """
    So sánh số cặp QA giữa output_data (đã có difficulty) và vqa_data (đã clean).
    - Nếu image_id có trong output nhưng KHÔNG còn trong vqa_data -> xóa khỏi output.
    - Nếu số lượng QA khác nhau -> xóa khỏi output để gen lại từ đầu.
    Trả về (output_data đã clean, danh sách các id bị xóa).
    """
    removed_mismatch = []
    removed_orphan = []

    # Duyệt qua snapshot keys vì sẽ mutate dict
    for image_id in list(output_data.keys()):
        if image_id not in vqa_data:
            # Image_id này đã bị xóa khỏi vqa cleaned -> orphan
            del output_data[image_id]
            removed_orphan.append(image_id)
            continue

        cleaned_qa = vqa_data[image_id]
        existing = output_data[image_id]

        # Nếu rỗng/None thì cũng coi như cần gen lại
        if not existing or not isinstance(existing, list):
            del output_data[image_id]
            removed_mismatch.append(image_id)
            continue

        if len(existing) != len(cleaned_qa):
            del output_data[image_id]
            removed_mismatch.append(image_id)

    if verbose:
        print(f"[*] Reconcile: removed {len(removed_mismatch)} mismatch, "
              f"{len(removed_orphan)} orphan entries.")
        if removed_mismatch[:5]:
            print(f"    Mismatch examples: {removed_mismatch[:5]}")
        if removed_orphan[:5]:
            print(f"    Orphan examples: {removed_orphan[:5]}")

    return output_data, removed_mismatch + removed_orphan


def process_difficulty_pipeline(args):
    # 1. Load input data
    print("[*] Loading input data...")
    with open(args.database_path, "r", encoding="utf-8") as f:
        database = json.load(f)
    with open(args.caption_json_path, "r", encoding="utf-8") as f:
        caption_data = json.load(f)
    with open(args.vqa_json_path, "r", encoding="utf-8") as f:
        vqa_data = json.load(f)

    # 2. Resume: load output dict hiện tại
    output_data = {}
    if os.path.exists(args.output_json_path):
        try:
            with open(args.output_json_path, "r", encoding="utf-8") as f:
                output_data = json.load(f)
            print(f"[*] Resuming: {len(output_data)} images already processed.")
        except Exception as e:
            print(f"[!] Could not load existing JSON: {e}")
            output_data = {}

    # 2.5. Reconcile với vqa đã clean: xóa các entry mismatch / orphan
    before_count = len(output_data)
    output_data, removed_ids = reconcile_with_cleaned_vqa(output_data, vqa_data, verbose=True)
    after_count = len(output_data)
    if before_count != after_count:
        print(f"[*] Output reduced from {before_count} -> {after_count} entries.")
        # Lưu ngay sau khi reconcile để file trên đĩa cũng sạch
        save_json(output_data, args.output_json_path)
        print(f"[*] Saved cleaned output to {args.output_json_path}")

    # 3. Tạo task list - giờ logic đơn giản: chỉ cần xét những id chưa có trong output_data
    all_tasks = []
    for image_id, qa_list in vqa_data.items():
        if image_id in output_data and output_data[image_id]:
            # Đã có và đã được reconcile -> chắc chắn match -> skip
            continue

        entry = caption_data.get(image_id)
        image_path = find_image_path(args.image_dir, image_id)

        if not entry:
            print(f"[!] Không tìm thấy caption cho {image_id}, bỏ qua.")
            continue
        if not image_path:
            print(f"[!] Không tìm thấy ảnh cho {image_id}, bỏ qua.")
            continue

        article_id = entry.get("article_id", image_id)
        article_content = database.get(article_id, {}).get("content", "Không có nội dung.")
        prompt = render_difficulty_prompt(args.template_dir, article_content, qa_list)

        all_tasks.append({
            "image_id": image_id,
            "image_path": image_path,
            "prompt": prompt,
            "prompt_len": len(prompt),
            "qa_list": qa_list
        })

    if not all_tasks:
        print("[*] No new tasks to process.")
        return

    all_tasks.sort(key=lambda x: x['prompt_len'], reverse=args.reverse_sort)
    if args.limit and args.limit > 0:
        all_tasks = all_tasks[:args.limit]

    backup_path = args.output_json_path.replace(".json", ".jsonl")
    print(f"[*] Total tasks: {len(all_tasks)}")
    print(f"[*]   - Re-gen (mismatch/orphan):  {sum(1 for t in all_tasks if t['image_id'] in removed_ids)}")
    print(f"[*]   - Brand new (never processed): {sum(1 for t in all_tasks if t['image_id'] not in removed_ids)}")
    print(f"[*] Backup file: {backup_path}")

    # 4. Khởi tạo model
    print(f"[*] Initializing model: {args.model_name} on {args.device}")
    model = CustomQwenVLCaptionModel(model_name=args.model_name, device=args.device)

    # 5. Batch processing với dual-write
    updated_count = 0
    pbar = tqdm(total=len(all_tasks), desc="Difficulty Rating")

    try:
        with open(backup_path, "a", encoding="utf-8") as f_backup:
            for i in range(0, len(all_tasks), args.batch_size):
                batch = all_tasks[i: i + args.batch_size]
                batch_prompts   = [t['prompt']     for t in batch]
                batch_img_paths = [t['image_path'] for t in batch]
                batch_ids       = [t['image_id']   for t in batch]

                t1 = time.perf_counter()
                try:
                    batch_results = model.generate_difficulty_batch(
                        image_paths=batch_img_paths,
                        prompts=batch_prompts
                    )
                    t2 = time.perf_counter()

                    for idx, result in enumerate(batch_results):
                        img_id      = batch[idx]["image_id"]
                        original_qa = batch[idx]["qa_list"]

                        print(result)

                        if result and isinstance(result, list) and len(result) == len(original_qa):
                            updated_qa = [
                                list(qa_pair[:2]) + [str(result[qa_idx])]
                                for qa_idx, qa_pair in enumerate(original_qa)
                            ]
                        else:
                            print(f"\n[!] Kết quả không hợp lệ cho {img_id}, ghi QA gốc.")
                            updated_qa = original_qa

                        # Ghi vào RAM dict
                        output_data[img_id] = updated_qa
                        updated_count += 1

                        # Ghi ngay vào backup .jsonl
                        f_backup.write(json.dumps({img_id: updated_qa}, ensure_ascii=False) + "\n")

                    f_backup.flush()

                    if i % (args.batch_size * 2) == 0:
                        print(f"\n⏱️ Batch {i//args.batch_size}: {t2-t1:.2f}s | {(t2-t1)/len(batch):.2f}s/img")

                    # Ghi .json chính mỗi 5 batch
                    if (i // args.batch_size) % 5 == 0:
                        save_json(output_data, args.output_json_path)
                        if "cuda" in args.device:
                            torch.cuda.empty_cache()

                except Exception as e:
                    print(f"\n[!] Error in batch starting at {batch_ids[0]}: {e}")

                pbar.update(len(batch))

    except KeyboardInterrupt:
        print("\n[!] Interrupted. Saving current progress...")
    except Exception as e:
        print(f"\n[!] Unexpected error: {e}")
    finally:
        if updated_count > 0:
            save_json(output_data, args.output_json_path)
            print(f"\n[*] Processed {updated_count} new entries. Total: {len(output_data)}")
        pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path",     type=str, default="../Eventa/webCrawl/src/merged_7_database.json")
    parser.add_argument("--caption_json_path", type=str, default="./image_caption_updated.json")
    parser.add_argument("--vqa_json_path",     type=str, default="./image_vqa.json")
    parser.add_argument("--output_json_path",  type=str, default="./image_vqa_with_difficulty_cleaned.json")
    parser.add_argument("--image_dir",         type=str, default="../Eventa/webCrawl/src/database_image")
    parser.add_argument("--template_dir",      type=str, default="./prompt_templates")
    parser.add_argument("--model_name",        type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device",            type=str, default="cuda:6")
    parser.add_argument("--batch_size",        type=int, default=48)
    parser.add_argument("--limit",             type=int, default=None)
    parser.add_argument("--reverse_sort",      action="store_true")
    args = parser.parse_args()

    process_difficulty_pipeline(args)