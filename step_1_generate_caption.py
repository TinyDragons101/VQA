import json
import argparse
import os
import torch
import re
from PIL import Image
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from torch.utils.data import Dataset, DataLoader
from internvl import CustomQwenVLCaptionModel
import time

def estimate_len(x):
    print(f"Estimating length for image_id: {x['image_id']} - title: {len(x['title'])}, content: {len(x['content'])}, original_caption: {len(x['original_caption'])}")
    return len(x["title"]) + len(x["content"]) + len(x["original_caption"])

# --- Cải tiến 1: Sử dụng Dataset để tận dụng đa nhân CPU load ảnh và render prompt ---
class CaptionDataset(Dataset):
    def __init__(self, tasks, template_dir):
        self.tasks = tasks
        # Khởi tạo Jinja2 Environment một lần duy nhất
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True, 
            lstrip_blocks=True
        )
        self.template = self.env.get_template("caption_generating.j2")

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        t = self.tasks[idx]
        try:
            prompt = self.template.render(
                title=t['title'], 
                caption=t['original_caption'], 
                # content=t['content']
                content=t['content'][:5000] if len(t['content']) > 5000 else t['content']
            )
            image = Image.open(t['image_path']).convert("RGB")
            image = image.resize((504, 504))
            
            return {
                "image": image,
                "prompt": prompt,
                "image_id": t['image_id'],
                "article_id": t['article_id'],
                "original_caption": t['original_caption'],
                "title": t['title']
            }
        except Exception as e:
            return None

# --- Cải tiến 2: Hàm gom nhóm (collate) để tiền xử lý Batch trước khi đưa vào GPU ---
def collate_fn(batch, processor):
    batch = [item for item in batch if item is not None]
    if not batch: return None

    images = [item['image'] for item in batch]
    prompts = [item['prompt'] for item in batch]
    
    # Áp dụng Chat Template cho cả Batch
    texts = []
    for p in prompts:
        messages = [
            {
                "role": "system", 
                "content": "Bạn là một biên tập viên nội dung chuyên nghiệp, chuyên phân tích văn hóa và nghệ thuật Việt Nam."
            },
            {
                "role": "user", 
                "content": [{"type": "image"}, {"type": "text", "text": p}]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)

    # Processor xử lý pixel ảnh và tokenize text (CPU/GPU tùy config)
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    return inputs, batch

def process_pipeline(args):
    # 1. Kiểm tra môi trường
    if not os.path.exists(args.database_path):
        print(f"[Error] Database not found: {args.database_path}")
        return

    print(f"[*] Scanning image directory: {args.image_dir}")
    existing_files = set(os.listdir(args.image_dir))
    valid_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.PNG', '.JPG', '.JPEG', '.WEBP']

    # 2. Load database & Progress
    with open(args.database_path, "r", encoding="utf-8") as f:
        database = json.load(f)

    output_data = {}
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            output_data = json.load(f)
        print(f"[*] Resuming: {len(output_data)} entries loaded.")

    # 3. Lọc Task
    all_tasks = []
    for art_id, art in database.items():
        for img in art.get("images", []):
            image_id = img.get("image_id")
            if not image_id or (image_id in output_data and output_data[image_id].get("generated_caption")):
                continue

            for ext in valid_extensions:
                target_file = f"{image_id}{ext}"
                if target_file in existing_files:
                    all_tasks.append({
                        "image_id": image_id,
                        "image_path": os.path.join(args.image_dir, target_file),
                        "article_id": art_id,
                        "title": art.get("title", ""),
                        "content": art.get("content", ""),
                        "original_caption": img.get("caption", "")
                    })
                    break

    if not all_tasks:
        print("[*] No new images to process.")
        return

    all_tasks.sort(key=estimate_len)

    # 4. Init Model & DataLoader
    print(f"[*] Initializing Model on {args.device}...")
    model_wrapper = CustomQwenVLCaptionModel(model_name=args.model_name, device=args.device)
    
    dataset = CaptionDataset(all_tasks, args.template_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,  # Tăng tốc load ảnh bằng nhiều luồng CPU
        collate_fn=lambda x: collate_fn(x, model_wrapper.processor),
        pin_memory=True,                 # ⚡ tăng tốc transfer CPU → GPU
        persistent_workers=False,         # ⚡ tránh fork lại worker - Phải là False nếu num_workers=0
        prefetch_factor=None,               # ⚡ preload batch
    )

    # Đường dẫn file backup để ghi liên tục
    backup_path = args.output_path.replace(".json", ".jsonl")
    
    # 5. Loop xử lý chính
    print(f"[*] Starting Batch Processing (Batch Size: {args.batch_size})...")
    pbar = tqdm(total=len(all_tasks), desc="Total Progress")

    total_oom_count = 0 
    max_oom_allowed = 10
    
    try:
        # Mở file .jsonl để append kết quả ngay lập tức (tránh mất data khi crash)
        with open(backup_path, "a", encoding="utf-8") as f_backup:
            for batch_data in dataloader:
                if batch_data is None: continue

                t0 = time.perf_counter()
                
                inputs, batch_info = batch_data
                t1 = time.perf_counter()

                inputs = inputs.to(model_wrapper.device)
                t2 = time.perf_counter()
                
                try:
                    try_count = 0

                    with torch.no_grad():
                        output_ids = model_wrapper.model.generate(
                            **inputs,
                            max_new_tokens=300,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9
                        )
                    t3 = time.perf_counter()

                    # Giải mã kết quả
                    generated_ids = [out[len(ins):] for ins, out in zip(inputs.input_ids, output_ids)]
                    responses = model_wrapper.processor.batch_decode(
                        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    t4 = time.perf_counter()

                    for idx, resp in enumerate(responses):
                        t = batch_info[idx]
                        res_json = model_wrapper.extract_json_object(resp)
                        
                        final_entry = {
                            "article_id": t['article_id'],
                            "title": t['title'],
                            "category": res_json.get("category", "") if res_json else "",
                            "original_caption": t['original_caption'],
                            "generated_caption": res_json.get("caption", resp) if res_json else resp
                        }
                        
                        # Lưu vào memory và ghi file backup
                        output_data[t['image_id']] = final_entry
                        f_backup.write(json.dumps({t['image_id']: final_entry}, ensure_ascii=False) + "\n")
                        f_backup.flush() # Đảm bảo data được ghi xuống đĩa
                    t5 = time.perf_counter()

                    print(f"""
                    ⏱️ Timing:
                    - DataLoader + collate: {t1 - t0:.3f}s
                    - Move to GPU:         {t2 - t1:.3f}s
                    - Generate (GPU):      {t3 - t2:.3f}s
                    - Decode:              {t4 - t3:.3f}s
                    - Extract:             {t5 - t4:.3f}s
                    """)

                    pbar.update(len(batch_info))
                    
                except torch.cuda.OutOfMemoryError:
                    total_oom_count += 1
                    torch.cuda.empty_cache()
                    print(f"\n[!] OOM lần {total_oom_count}/{max_oom_allowed} tại batch hiện tại.")
                    
                    if total_oom_count >= max_oom_allowed:
                        print("!!! Quá giới hạn OOM cho phép. Đang dừng chương trình để bảo vệ GPU...")
                        raise  # Đẩy lỗi ra ngoài để rơi vào khối 'finally' lưu data
                    
                    continue

    except KeyboardInterrupt:
        print("\n[!] User Interrupted. Saving current progress...")
        
    except Exception as e:
        print(f"\n[!] Error: {e}")
        
    finally:
        # Lưu file JSON cuối cùng một cách an toàn
        os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        pbar.close()
        print(f"[*] Finished! Total saved: {len(output_data)} entries.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str, default="../Eventa/webCrawl/src/merged_2_database.json")
    parser.add_argument("--output_path", type=str, default="./image_caption_updated.json")
    parser.add_argument("--image_dir", type=str, default="../Eventa/webCrawl/src/database_image")
    parser.add_argument("--template_dir", type=str, default="./prompt_templates")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:6")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)

    process_pipeline(parser.parse_args())

if __name__ == "__main__":
    main()