import json
import argparse
import os
import torch
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from internvl import CustomQwenVLCaptionModel
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# Dùng Lock để tránh xung đột khi ghi file/update dict từ nhiều thread
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

def process_image(img_info, article_info, args, model, output_data):
    image_id = img_info.get("image_id")
    if not image_id:
        return False

    # Kiểm tra xem đã xử lý chưa (Thread-safe read vì output_data là dict)
    if image_id in output_data and output_data[image_id].get("generated_caption"):
        return False

    image_path = os.path.join(args.image_dir, f"{image_id}.png")
    if not os.path.exists(image_path):
        return False

    try:
        prompt = render_caption_prompt(
            args.template_dir, 
            article_info['title'], 
            img_info.get("caption", ""), 
            article_info['content']
        )
        
        # Inference (VLM thường tự quản lý CUDA stream bên trong)
        response = model.generate_caption_and_category(
            image_path, 
            prompt=prompt, 
            max_new_tokens=300, 
            do_sample=True
        )
        
        entry = {
            "article_id": article_info['id'],
            "title": article_info['title'],
            "category": response["category"],
            "original_caption": img_info.get("caption", ""),
            "generated_caption": response["caption"]
        }

        with save_lock:
            output_data[image_id] = entry
        return True

    except Exception as e:
        print(f"\n[!] Failed image {image_id}: {e}")
        return False

def process_pipeline(args):
    if not os.path.exists(args.database_path):
        print(f"[Error] Database not found: {args.database_path}")
        return

    # 1. Load database
    with open(args.database_path, "r", encoding="utf-8") as f:
        database = json.load(f)

    # 2. Load progress (Resume)
    output_data = {}
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            output_data = json.load(f)
        print(f"[*] Resuming: {len(output_data)} entries loaded.")

    # 3. Init Model
    print(f"[*] Initializing Model on {args.device}...")
    model = CustomQwenVLCaptionModel(model_name=args.model_name, device=args.device)

    # 4. Lọc các task chưa được xử lý
    all_tasks = []
    for art_id, art in database.items():
        for img in art.get("images", []):
            image_id = img.get("image_id")
            if not image_id: continue
            
            # Kiểm tra xem đã có trong output_data chưa
            if image_id in output_data and output_data[image_id].get("generated_caption"):
                continue

            image_path = os.path.join(args.image_dir, f"{image_id}.png")
            if os.path.exists(image_path):
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
    
    # 5. Xử lý theo từng Batch
    batch_size = args.batch_size
    updated_count = 0
    
    
    pbar = tqdm(total=len(all_tasks), desc="Batch Processing")
    
    for i in range(0, len(all_tasks), batch_size):
        batch = all_tasks[i : i + batch_size]
        
        # Chuẩn bị dữ liệu cho batch
        prompts = []
        img_paths = []
        for t in batch:
            p = render_caption_prompt(args.template_dir, t['title'], t['original_caption'], t['content'])
            prompts.append(p)
            img_paths.append(t['image_path'])

        try:
            # Inference toàn bộ batch cùng lúc
            # Giả định hàm trả về một list các dict [{'caption': '...', 'category': '...'}, ...]
            results = model.generate_batch(
                img_paths, 
                prompts=prompts, 
                max_new_tokens=300, 
                do_sample=True
            )

            # Lưu kết quả
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

            # Lưu file định kỳ
            if updated_count > 0 and updated_count % 40 == 0:
                save_json(output_data, args.output_path)
            
            # Giải phóng cache GPU để tránh phân mảnh bộ nhớ
            if updated_count % 100 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n[!] Failed processing batch at index {i}: {e}")
            # Trong trường hợp lỗi cả batch, bạn có thể xử lý retry từng ảnh ở đây nếu cần
        
        pbar.update(len(batch))

    pbar.close()
    
    # 6. Final Save
    save_json(output_data, args.output_path)
    print(f"\n[Summary] Total entries: {len(output_data)} | Newly Generated: {updated_count}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str, default="../Eventa/webCrawl/src/database.json")
    parser.add_argument("--output_path", type=str, default="./image_caption.json")
    parser.add_argument("--image_dir", type=str, default="../Eventa/webCrawl/src/database_image")
    parser.add_argument("--template_dir", type=str, default="./prompt_templates")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4, help="Số luồng chạy song song")
    parser.add_argument("--batch_size", type=int, default=12)

    
    process_pipeline(parser.parse_args())

if __name__ == "__main__":
    main()