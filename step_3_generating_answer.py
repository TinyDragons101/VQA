import json
import argparse
import os
import torch
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from internvl import CustomQwenVLCaptionModel

def render_answer_prompt(template_dir, entry, article_content, questions):
    """
    Render prompt với danh sách câu hỏi sử dụng Jinja2.
    """
    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True
    )
    template = env.get_template("answer_generating.j2")
    return template.render(
        title=entry.get("title", ""),
        content=article_content[:3000], # Giới hạn content để tránh quá tải context
        original_caption=entry.get("original_caption", ""),
        generated_caption=entry.get("generated_caption", ""),
        questions=questions
    )

def save_json(data, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def find_image_path(image_dir, image_id):
    for ext in ['.png', '.jpg', '.jpeg', '.JPG', '.PNG']:
        path = os.path.join(image_dir, f"{image_id}{ext}")
        if os.path.exists(path):
            return path
    return None

def process_answering_pipeline(args):
    # 1. Load dữ liệu
    print("[*] Loading data files...")
    with open(args.database_path, "r", encoding="utf-8") as f:
        database = json.load(f)
    
    with open(args.caption_json_path, "r", encoding="utf-8") as f:
        caption_data = json.load(f)

    if not os.path.exists(args.vqa_json_path):
        print(f"[Error] VQA questions file not found: {args.vqa_json_path}")
        return
        
    with open(args.vqa_json_path, "r", encoding="utf-8") as f:
        vqa_data = json.load(f)

    # 2. Thu thập các task chưa hoàn thành
    all_tasks = []
    for image_id, vqa_list in vqa_data.items():
        # Kiểm tra xem ảnh này có câu hỏi nào chưa có câu trả lời không
        unanswered_indices = [i for i, item in enumerate(vqa_list) if not item.get("answer")]
        
        if not unanswered_indices:
            continue

        entry = caption_data.get(image_id)
        image_path = find_image_path(args.image_dir, image_id)
        
        if entry and image_path:
            article_id = entry.get("article_id")
            article_content = database.get(article_id, {}).get("content", "Không có nội dung bài báo.")
            questions_to_ask = [vqa_list[i]["question"] for i in unanswered_indices]
            
            all_tasks.append({
                "image_id": image_id,
                "image_path": image_path,
                "entry": entry,
                "article_content": article_content,
                "questions_to_ask": questions_to_ask,
                "unanswered_indices": unanswered_indices
            })

    if not all_tasks:
        print("[*] All questions are already answered.")
        return

    # 3. Khởi tạo Model
    print(f"[*] Initializing model: {args.model_name} on {args.device}")
    model = CustomQwenVLCaptionModel(model_name=args.model_name, device=args.device)

    # 4. Xử lý theo Batch
    batch_size = args.batch_size
    updated_count = 0
    pbar = tqdm(total=len(all_tasks), desc="Batch Answering")

    for i in range(0, len(all_tasks), batch_size):
        batch = all_tasks[i : i + batch_size]
        
        batch_prompts = []
        batch_img_paths = []

        for t in batch:
            prompt = render_answer_prompt(
                args.template_dir, 
                t['entry'], 
                t['article_content'], 
                t['questions_to_ask']
            )
            batch_prompts.append(prompt)
            batch_img_paths.append(t['image_path'])

        try:
            # Gọi hàm inference batch (Bạn cần thêm hàm generate_answers_batch vào class model tương tự generate_questions_batch)
            # Hàm này trả về: [[ans1, ans2, ...], [ans1, ans2, ...], ...]
            batch_results = model.generate_answers_batch(
                batch_img_paths, 
                prompts=batch_prompts
            )

            for idx, answers in enumerate(batch_results):
                task = batch[idx]
                image_id = task['image_id']
                
                if answers and isinstance(answers, list) and len(answers) == len(task['questions_to_ask']):
                    # Cập nhật vào dữ liệu gốc vqa_data
                    for q_idx_in_task, actual_ans in zip(task['unanswered_indices'], answers):
                        vqa_data[image_id][q_idx_in_task]["answer"] = str(actual_ans).strip()
                    updated_count += 1
                else:
                    print(f"\n[!] Mismatch in answers count for {image_id}")

            # Lưu định kỳ
            if updated_count > 0 and updated_count % 20 == 0:
                save_json(vqa_data, args.vqa_json_path)
            
            if updated_count % 50 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n[!] Failed processing batch at index {i}: {e}")
        
        pbar.update(len(batch))

    pbar.close()

    # 5. Lưu kết quả cuối cùng
    save_json(vqa_data, args.vqa_json_path)
    print(f"\n[Summary] Total images updated: {updated_count}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str, default="../Eventa/webCrawl/src/database.json")
    parser.add_argument("--caption_json_path", type=str, default="./image_caption.json")
    parser.add_argument("--vqa_json_path", type=str, default="./image_vqa.json")
    parser.add_argument("--image_dir", type=str, default="../Eventa/webCrawl/src/database_image")
    parser.add_argument("--template_dir", type=str, default="./prompt_templates")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=12)
    
    args = parser.parse_args()
    process_answering_pipeline(args)

if __name__ == "__main__":
    main()