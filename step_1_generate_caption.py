import json
import argparse
import os
import torch
import sys
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from internvl import CustonInternVLCaptionModel, CustomQwenVLCaptionModel, CustomInternVLCaptionModel14B

def render_caption_prompt(template_dir, title, caption, content):
    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True
    )
    template = env.get_template("caption_generating.j2")
    return template.render(title=title, caption=caption, content=content)

def save_json(data, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def process_pipeline(args):
    if not os.path.exists(args.database_path):
        print(f"[Error] Database not found: {args.database_path}")
        return

    with open(args.database_path, "r", encoding="utf-8") as f:
        database = json.load(f)

    output_data = {}
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            output_data = json.load(f)
        print(f"[*] Resuming: {len(output_data)} entries loaded.")

    model = None
    if args.generate_caption:
        print(f"[*] Initializing InternVL on {args.device}...")
        model = CustomQwenVLCaptionModel(model_name=args.model_name, device=args.device)

    updated_count = 0
    try:
        for article_id, article in tqdm(database.items(), desc="Processing"):
            title = article.get("title", "")
            content = article.get("content", "")
            images = article.get("images", [])

            for img in images:
                image_id = img.get("image_id")
                if not image_id: continue

                # Skip if already processed
                if image_id in output_data and output_data[image_id].get("generated_caption"):
                    continue

                image_path = os.path.join(args.image_dir, f"{image_id}.png")
                
                # Metadata entry
                entry = {
                    "article_id": article_id,
                    "title": title,
                    "category": "",
                    "original_caption": img.get("caption", ""),
                    "generated_caption": ""
                }

                if args.generate_caption and model and os.path.exists(image_path):
                    try:
                        prompt = render_caption_prompt(args.template_dir, title, entry["original_caption"], content)
                        
                        response = model.generate_caption_and_category(
                            image_path, 
                            prompt=prompt, 
                            max_new_tokens=300, 
                            do_sample=True
                        )
                        
                        entry["generated_caption"] = response["caption"]
                        entry["category"] = response["category"]
                        updated_count += 1
                        
                        # Memory Cleanup
                        # del image_tensor
                        if updated_count % 50 == 0:
                            torch.cuda.empty_cache()

                    except Exception as e:
                        print(f"\n[!] Failed image {image_id}: {e}")

                output_data[image_id] = entry

                if updated_count > 0 and updated_count % 20 == 0:
                    save_json(output_data, args.output_path)

    except KeyboardInterrupt:
        print("\n[!] Process interrupted by user. Saving progress...")
    finally:
        save_json(output_data, args.output_path)
        print(f"\n[Summary] Total: {len(output_data)} | Newly Generated: {updated_count}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str, default="./database.json")
    parser.add_argument("--output_path", type=str, default="./image_caption.json")
    parser.add_argument("--image_dir", type=str, default="./images")
    parser.add_argument("--template_dir", type=str, default="./prompt_templates")
    parser.add_argument("--generate_caption", action="store_true", default=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    process_pipeline(parser.parse_args())

if __name__ == "__main__":
    main()