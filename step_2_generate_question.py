import json
import argparse
import os
import torch
import re
from tqdm import tqdm
from PIL import Image
from jinja2 import Environment, FileSystemLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, GenerationConfig
from internvl import CustomQwenVLCaptionModel

def render_vqa_prompt(template_dir, entry):
    env = Environment(loader=FileSystemLoader(template_dir))
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
    # Load kết quả từ Phase 1
    if not os.path.exists(args.caption_json_path):
        print(f"[Error] Phase 1 output not found: {args.caption_json_path}")
        return

    with open(args.caption_json_path, "r", encoding="utf-8") as f:
        caption_data = json.load(f)

    vqa_output = {}
    if os.path.exists(args.output_vqa_path):
        with open(args.output_vqa_path, "r", encoding="utf-8") as f:
            vqa_output = json.load(f)
        print(f"[*] Resuming: {len(vqa_output)} images already have questions.")

    model = CustomQwenVLCaptionModel(model_name=args.model_name, device=args.device)

    updated_count = 0
    try:
        for image_id, entry in tqdm(caption_data.items(), desc="Generating Questions"):
            if image_id in vqa_output:
                continue

            image_path = os.path.join(args.image_dir, f"{image_id}.png")
            if not os.path.exists(image_path):
                continue

            try:
                prompt = render_vqa_prompt(args.template_dir, entry)
                questions = model.generate_questions(image_path, prompt)

                if questions and isinstance(questions, list):
                    # Format theo yêu cầu: [{question: ..., answer: ""}, ...]
                    vqa_list = [{"question": q, "answer": ""} for q in questions[:5]]
                    vqa_output[image_id] = vqa_list
                    updated_count += 1
                
                if updated_count % 10 == 0:
                    save_json(vqa_output, args.output_vqa_path)
                    if updated_count % 50 == 0:
                        torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n[!] Error at {image_id}: {e}")

    except KeyboardInterrupt:
        print("\n[!] Interrupted. Saving...")
    finally:
        save_json(vqa_output, args.output_vqa_path)
        print(f"完成! Total images with VQA: {len(vqa_output)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption_json_path", type=str, default="./image_caption.json")
    parser.add_argument("--output_vqa_path", type=str, default="./image_vqa.json")
    parser.add_argument("--image_dir", type=str, default="./images")
    parser.add_argument("--template_dir", type=str, default="./prompt_templates")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    
    args = parser.parse_args()
    process_vqa_pipeline(args)

if __name__ == "__main__":
    main()