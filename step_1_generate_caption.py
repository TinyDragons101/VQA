import json
import argparse
import os
import torch
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from internvl import CustomQwenVLCaptionModel
from threading import Lock

# Dùng Lock để tránh xung đột khi ghi file
save_lock = Lock()

def render_caption_prompt(template_dir, title, caption, content):
    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True
    )
    template = env.get_template("caption_generating.j2")
    return template.render(title=title, caption=caption, content=content)

def save_json(data, path):
    with save_lock:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

def process_pipeline(args):
    if not os.path.exists(args.database_path):
        print(f"[Error] Database not found: {args.database_path}")
        return

    # 1. Quét toàn bộ file trong thư mục ảnh để tra cứu cực nhanh (O(1))
    print(f"[*] Scanning image directory: {args.image_dir}")
    if not os.path.exists(args.image_dir):
        print(f"[Error] Image directory not found: {args.image_dir}")
        return
    
    # Lưu vào set để check file tồn tại trong tích tắc
    existing_files = set(os.listdir(args.image_dir))
    valid_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.PNG', '.JPG', '.JPEG', '.WEBP']

    # 2. Load database
    with open(args.database_path, "r", encoding="utf-8") as f:
        database = json.load(f)

    # 3. Load progress (Resume)
    output_data = {}
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            output_data = json.load(f)
        print(f"[*] Resuming: {len(output_data)} entries loaded from output file.")

    # 4. Lọc các task chưa được xử lý và kiểm tra file ảnh tồn tại
    all_tasks = []
    print("[*] Filtering tasks and checking image extensions...")
    for art_id, art in database.items():
        for img in art.get("images", []):
            image_id = img.get("image_id")
            if not image_id: 
                continue
            
            # Bỏ qua nếu đã xử lý rồi
            if image_id in output_data and output_data[image_id].get("generated_caption"):
                continue

            # Kiểm tra xem file ảnh với đuôi nào đang tồn tại trong set
            found_filename = None
            for ext in valid_extensions:
                target_file = f"{image_id}{ext}"
                if target_file in existing_files:
                    found_filename = target_file
                    break
            
            if found_filename:
                image_path = os.path.join(args.image_dir, found_filename)
                all_tasks.append({
                    "image_id": image_id,
                    "image_path": image_path,
                    "article_id": art_id,
                    "title": art.get("title", ""),
                    "content": art.get("content", ""),
                    "original_caption": img.get("caption", "")
                })

    if not all_tasks:
        print("[*] No new images to process.")
        return

    print(f"[*] Total tasks to process: {len(all_tasks)}")

    # 5. Init Model
    print(f"[*] Initializing Model on {args.device}...")
    model = CustomQwenVLCaptionModel(model_name=args.model_name, device=args.device)
    
    # 6. Xử lý theo từng Batch
    batch_size = args.batch_size
    updated_count = 0
    
    pbar = tqdm(total=len(all_tasks), desc="Batch Processing")
    
    for i in range(0, len(all_tasks), batch_size):
        batch = all_tasks[i : i + batch_size]
        
        prompts = []
        img_paths = []
        for t in batch:
            p = render_caption_prompt(args.template_dir, t['title'], t['original_caption'], t['content'])
            prompts.append(p)
            img_paths.append(t['image_path'])

        try:
            results = model.generate_batch(
                img_paths, 
                prompts=prompts, 
                max_new_tokens=300, 
                do_sample=True
            )

            for idx, res in enumerate(results):
                t = batch[idx]
                output_data[t['image_id']] = {
                    "article_id": t['article_id'],
                    "title": t['title'],
                    "category": res.get("category", ""),
                    "original_caption": t['original_caption'],
                    "generated_caption": res.get("caption", "")
                }
                updated_count += 1

            # Lưu file định kỳ (mỗi 120 ảnh)
            if updated_count > 0 and updated_count % 120 == 0:
                save_json(output_data, args.output_path)
            
            # Giải phóng cache GPU (mỗi 1000 ảnh)
            if updated_count % 1000 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n[!] Failed processing batch at index {i}: {e}")
        
        pbar.update(len(batch))

    pbar.close()
    
    # 7. Final Save
    save_json(output_data, args.output_path)
    print(f"\n[Summary] Total entries in file: {len(output_data)} | Newly Generated: {updated_count}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str, default="../Eventa/webCrawl/src/merged_database.json")
    parser.add_argument("--output_path", type=str, default="./image_caption.json")
    parser.add_argument("--image_dir", type=str, default="../Eventa/webCrawl/src/database_image")
    parser.add_argument("--template_dir", type=str, default="./prompt_templates")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:4")
    parser.add_argument("--batch_size", type=int, default=16)

    process_pipeline(parser.parse_args())

if __name__ == "__main__":
    main()