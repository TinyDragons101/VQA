import os
import json
import argparse
import time
import torch
from typing import List, Dict, Any
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from internvl import CustomQwenVLCaptionModel 

def render_qa_prompt(template_dir, entry, article_content):
    """Render prompt yêu cầu model sinh cặp QA dựa trên ảnh và bài báo"""
    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True
    )
    # Bạn nên tạo file mới: qa_generating.j2
    template = env.get_template("answer_generating.j2")
    
    truncated_content = article_content[:5000] 
    
    return template.render(
        title=entry.get("title", ""),
        content=truncated_content,
        original_caption=entry.get("original_caption", ""),
        generated_caption=entry.get("generated_caption", "")
    )

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

def process_qa_generation_pipeline(args):
    
    # 1. Load Data
    print("[*] Loading data files...")
    with open(args.database_path, "r", encoding="utf-8") as f:
        database = json.load(f)
    with open(args.caption_json_path, "r", encoding="utf-8") as f:
        caption_data = json.load(f)

    # 2. Resume logic & Cleanup
    output_data = {}
    if os.path.exists(args.output_json_path):
        try:
            with open(args.output_json_path, "r", encoding="utf-8") as f:
                output_data = json.load(f)
            
            # --- BẮT ĐẦU LOGIC KIỂM TRA VÀ XÓA ---
            print("[*] Checking for obsolete entries in output...")
            original_count = len(output_data)
            
            # Tạo danh sách các id cần xóa để tránh lỗi "RuntimeError: dictionary changed size during iteration"
            ids_to_remove = []
            for img_id in output_data.keys():
                # Lấy article_id từ caption_data để check trong database
                # Nếu không tìm thấy img_id trong caption_data hoặc article_id không có trong database thì xóa
                entry = caption_data.get(img_id, {})
                article_id = entry.get("article_id", img_id)
                
                if article_id not in database:
                    ids_to_remove.append(img_id)
            
            if ids_to_remove:
                for img_id in ids_to_remove:
                    del output_data[img_id]
                
                print(f"[!] Removed {len(ids_to_remove)} obsolete entries from output. ({original_count} -> {len(output_data)})")
                # I want to print some of the removed ids for debugging purposes
                print(f"Sample removed ids: {ids_to_remove[:10]}")
                
                # Lưu lại file output ngay sau khi dọn dẹp để đồng bộ
                save_json(output_data, args.output_json_path)
            else:
                print("[*] No obsolete entries found.")
            # --- KẾT THÚC LOGIC KIỂM TRA ---

            print(f"[*] Resuming: {len(output_data)} images already processed.")
        except Exception as e:
            print(f"[!] Error during resume/cleanup: {e}")

    # 3. Chuẩn bị Task (Lặp qua caption_data thay vì vqa_data)
    all_tasks = []
    for image_id, entry in caption_data.items():
        # Kiểm tra nếu đã xử lý rồi thì bỏ qua
        if image_id in output_data and output_data[image_id]:
            continue

        image_path = find_image_path(args.image_dir, image_id)
        if image_path:
            article_id = entry.get("article_id", image_id) 
            article_info = database.get(article_id, {})
            article_content = article_info.get("content", "Không có nội dung.")
            
            # Render prompt mới không cần biến 'questions' truyền vào
            prompt = render_qa_prompt(args.template_dir, entry, article_content)
            
            all_tasks.append({
                "image_id": image_id,
                "image_path": image_path,
                "prompt": prompt,
                "prompt_len": len(prompt)
            })

    if not all_tasks:
        print("[*] No new images to process.")
        return
    
    all_tasks.sort(key=lambda x: x['prompt_len'], reverse=args.reverse_sort)
    if args.limit:
        all_tasks = all_tasks[:args.limit]

    backup_path = args.output_json_path.replace(".json", ".jsonl")
    print(f"[*] Total tasks: {len(all_tasks)} | Backup: {backup_path}")

    # 4. Khởi tạo Model
    model = CustomQwenVLCaptionModel(model_name=args.model_name, device=args.device)

    # 5. Batch Processing
    batch_size = args.batch_size
    updated_count = 0
    pbar = tqdm(total=len(all_tasks), desc="Qwen QA Generating")

    try:
        with open(backup_path, "a", encoding="utf-8") as f_backup:
            for i in range(0, len(all_tasks), batch_size):
                batch = all_tasks[i : i + batch_size]
                batch_prompts = [t['prompt'] for t in batch]
                batch_img_paths = [t['image_path'] for t in batch]
                batch_ids = [t['image_id'] for t in batch]
                
                try:
                    # Model sinh ra nội dung (kỳ vọng là chuỗi format JSON hoặc list)
                    batch_results = model.generate_answers_batch(
                        image_paths=batch_img_paths, 
                        prompts=batch_prompts
                    )
                    
                    for idx, result in enumerate(batch_results):
                        img_id = batch_ids[idx]
                        if result:
                            # Lưu kết quả thô hoặc parse nếu model trả về string JSON
                            output_data[img_id] = result 
                            updated_count += 1
                            f_backup.write(json.dumps({img_id: result}, ensure_ascii=False) + "\n")
                    
                    f_backup.flush()

                    if (i // batch_size) % 5 == 0:
                        save_json(output_data, args.output_json_path)
                        if "cuda" in args.device:
                            torch.cuda.empty_cache()

                except Exception as e:
                    print(f"\n[!] Error in batch {batch_ids[0]}: {e}")
                
                pbar.update(len(batch))

    except KeyboardInterrupt:
        print("\n[!] Interrupted. Saving...")
    finally:
        if updated_count > 0:
            save_json(output_data, args.output_json_path)
        pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str, default="../Eventa/webCrawl/src/merged_7_database.json")
    parser.add_argument("--caption_json_path", type=str, default="./image_caption_updated.json")
    parser.add_argument("--output_json_path", type=str, default="./image_vqa.json")
    parser.add_argument("--image_dir", type=str, default="../Eventa/webCrawl/src/database_image")
    parser.add_argument("--template_dir", type=str, default="./prompt_templates")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:4")
    parser.add_argument("--batch_size", type=int, default=64) # Giảm batch size vì sinh QA dài hơn trả lời
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reverse_sort", action="store_true")
    
    args = parser.parse_args()
    process_qa_generation_pipeline(args)