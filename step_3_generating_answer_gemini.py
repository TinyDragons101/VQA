import os
import json
import argparse
import time
from typing import List, Dict, Any
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from dotenv import load_dotenv
from gemini import GeminiVLCaptionModel

# Load môi trường (API Key)
load_dotenv()

def render_answer_prompt(template_dir, entry, article_content, questions):
    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True
    )
    template = env.get_template("answer_generating_gemini.j2")
    
    # Giới hạn nội dung bài báo để tránh tràn token nếu bài quá dài
    truncated_content = article_content[:5000] 
    
    return template.render(
        title=entry.get("title", ""),
        content=truncated_content,
        original_caption=entry.get("original_caption", ""),
        generated_caption=entry.get("generated_caption", ""),
        questions=questions
    )

def save_json(data, path):
    """Lưu file JSON an toàn bằng cách ghi vào file tạm trước"""
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

def process_answering_pipeline(args):
    # 1. Load Data
    print("[*] Loading data files...")
    if not os.path.exists(args.vqa_json_path):
        print(f"[Error] VQA input file not found: {args.vqa_json_path}")
        return

    with open(args.database_path, "r", encoding="utf-8") as f:
        database = json.load(f)
    with open(args.caption_json_path, "r", encoding="utf-8") as f:
        caption_data = json.load(f)
    with open(args.vqa_json_path, "r", encoding="utf-8") as f:
        vqa_data = json.load(f)

    # 2. Resume logic: Load output cũ
    output_data = {}
    if os.path.exists(args.output_json_path):
        try:
            with open(args.output_json_path, "r", encoding="utf-8") as f:
                output_data = json.load(f)
            print(f"[*] Resuming: {len(output_data)} images already answered.")
        except:
            output_data = {}

    # 3. Chuẩn bị Task
    all_tasks = []
    for image_id, questions in vqa_data.items():
        if image_id in output_data and output_data[image_id]:
            continue

        entry = caption_data.get(image_id)
        image_path = find_image_path(args.image_dir, image_id)
        
        if entry and image_path:
            article_id = entry.get("article_id", image_id) 
            article_info = database.get(article_id, {})
            article_content = article_info.get("content", "Không có nội dung.")
            
            prompt = render_answer_prompt(args.template_dir, entry, article_content, questions)
            
            all_tasks.append({
                "image_id": image_id,
                "image_path": image_path,
                "prompt": prompt,
                "prompt_len": len(prompt)
            })

    if not all_tasks:
        print("[*] No new images/questions to process.")
        return
    
    # Sắp xếp theo độ dài prompt để tối ưu hóa xử lý batch (tùy chọn)
    all_tasks.sort(key=lambda x: x['prompt_len'])
    
    if args.limit and args.limit > 0:
        all_tasks = all_tasks[:args.limit]
        print(f"[*] Limit applied: Processing {len(all_tasks)} tasks.")

    backup_path = args.output_json_path.replace(".json", ".jsonl")
    print(f"[*] Total tasks to process: {len(all_tasks)}")

    # 4. Khởi tạo Model
    model = GeminiVLCaptionModel(model_name=args.model_name)

    # 5. Batch Processing
    batch_size = args.batch_size
    updated_count = 0
    pbar = tqdm(total=len(all_tasks), desc="Gemini Answering")

    try:
        # Mở file backup .jsonl ở chế độ append
        with open(backup_path, "a", encoding="utf-8") as f_backup:
            for i in range(0, len(all_tasks), batch_size):
                batch = all_tasks[i : i + batch_size]
                batch_ids = [t['image_id'] for t in batch]
                
                for task in batch:
                    print(f"[*] Image ID: {task['image_id']} | Prompt Length: {task['prompt_len']} characters")

                try:
                    # Gọi API song song
                    batch_results = model.generate_answers_batch(
                        tasks=batch, 
                        max_workers=args.workers
                    )

                    for idx, result in enumerate(batch_results):
                        img_id = batch_ids[idx]
                        if result and isinstance(result, list):
                            # Lưu vào RAM
                            output_data[img_id] = result
                            updated_count += 1
                            
                            # Ghi ngay vào file backup .jsonl (mỗi dòng 1 object)
                            line = json.dumps({img_id: result}, ensure_ascii=False)
                            f_backup.write(line + "\n")
                    
                    # Đảm bảo dữ liệu được ghi xuống đĩa cứng
                    f_backup.flush()

                    # Lưu file .json chính định kỳ mỗi 5 batch
                    if (i // batch_size) % 5 == 0:
                        save_json(output_data, args.output_json_path)

                except Exception as e:
                    print(f"\n[!] Error at batch starting with {batch_ids[0]}: {e}")
                    # Tùy chọn: time.sleep(2) nếu cần
                
                pbar.update(len(batch))

    except KeyboardInterrupt:
        print("\n[!] User Interrupted (^C). Saving progress...")
    except Exception as e:
        print(f"\n[!] Unexpected error: {e}")
    finally:
        # Final Save: Luôn chạy khi kết thúc hoặc lỗi
        if updated_count > 0:
            save_json(output_data, args.output_json_path)
            print(f"\n[*] Final save completed. Total entries: {len(output_data)}")
        pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str, default="../Eventa/webCrawl/src/merged_database.json")
    parser.add_argument("--caption_json_path", type=str, default="./image_caption_updated.json")
    parser.add_argument("--vqa_json_path", type=str, default="./image_questions.json")
    parser.add_argument("--output_json_path", type=str, default="./image_vqa.json")
    parser.add_argument("--image_dir", type=str, default="../Eventa/webCrawl/src/database_image")
    parser.add_argument("--template_dir", type=str, default="./prompt_templates")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash-lite")
    
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--workers", type=int, default=5, help="Số luồng API chạy song song")
    parser.add_argument("--limit", type=int, default=1000, help="Giới hạn số lượng ảnh cần process")
    
    args = parser.parse_args()
    process_answering_pipeline(args)