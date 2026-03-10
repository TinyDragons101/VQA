import json
import argparse
import os
import torch
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from internvl import CustomQwenVLCaptionModel

def render_vqa_prompt(template_dir, entry):
    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True
    )
    template = env.get_template("question_generating.j2")
    return template.render(
        title=entry.get("title", ""),
        original_caption=entry.get("original_caption", ""),
        generated_caption=entry.get("generated_caption", ""),
        category=entry.get("category", "")
    )

def save_json(data, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def process_vqa_pipeline(args):
    # 1. Load kết quả từ Phase 1
    if not os.path.exists(args.caption_json_path):
        print(f"[Error] Phase 1 output not found: {args.caption_json_path}")
        return

    with open(args.caption_json_path, "r", encoding="utf-8") as f:
        caption_data = json.load(f)

    # 2. Load progress (Resume)
    vqa_output = {}
    if os.path.exists(args.output_vqa_path):
        with open(args.output_vqa_path, "r", encoding="utf-8") as f:
            vqa_output = json.load(f)
        print(f"[*] Resuming: {len(vqa_output)} images already processed.")

    # 3. Init Model
    print(f"[*] Initializing Model on {args.device}...")
    model = CustomQwenVLCaptionModel(model_name=args.model_name, device=args.device)

    # 4. Lọc các task chưa xử lý
    all_tasks = []
    for image_id, entry in caption_data.items():
        if image_id in vqa_output:
            continue
        
        image_path = os.path.join(args.image_dir, f"{image_id}.png")
        if os.path.exists(image_path):
            all_tasks.append({
                "image_id": image_id,
                "image_path": image_path,
                "entry": entry
            })

    if not all_tasks:
        print("[*] No new images to process.")
        return

    print(f"[*] Total tasks to process: {len(all_tasks)}")

    # 5. Xử lý theo Batch
    batch_size = args.batch_size
    updated_count = 0
    pbar = tqdm(total=len(all_tasks), desc="VQA Batch Processing")

    for i in range(0, len(all_tasks), batch_size):
        batch = all_tasks[i : i + batch_size]
        
        batch_prompts = []
        batch_img_paths = []
        batch_ids = []

        for t in batch:
            prompt = render_vqa_prompt(args.template_dir, t['entry'])
            batch_prompts.append(prompt)
            batch_img_paths.append(t['image_path'])
            batch_ids.append(t['image_id'])

        try:
            # Giả định hàm generate_questions_batch trả về list của list các câu hỏi
            # Ví dũ: [["Q1?", "Q2?"], ["Q3?", "Q4?"], ...]
            batch_results = model.generate_questions_batch(
                batch_img_paths, 
                prompts=batch_prompts,
                max_new_tokens=500
            )

            for idx, questions in enumerate(batch_results):
                img_id = batch_ids[idx]
                if questions and isinstance(questions, list):
                    # Format: [{question: ..., answer: ""}, ...]
                    vqa_list = [{"question": q, "answer": ""} for q in questions[:5]]
                    vqa_output[img_id] = vqa_list
                    updated_count += 1

            # Lưu định kỳ
            if updated_count > 0 and updated_count % 50 == 0:
                save_json(vqa_output, args.output_vqa_path)
            
            # Giải phóng cache GPU
            if updated_count % 100 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n[!] Failed processing batch at index {i}: {e}")
        
        pbar.update(len(batch))

    pbar.close()

    # 6. Final Save
    save_json(vqa_output, args.output_vqa_path)
    print(f"\n[Summary] Total VQA entries: {len(vqa_output)} | Newly Generated: {updated_count}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption_json_path", type=str, default="./image_caption.json")
    parser.add_argument("--output_vqa_path", type=str, default="./image_vqa.json")
    parser.add_argument("--image_dir", type=str, default="../Eventa/webCrawl/src/database_image")
    parser.add_argument("--template_dir", type=str, default="./prompt_templates")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    
    args = parser.parse_args()
    process_vqa_pipeline(args)

if __name__ == "__main__":
    main()