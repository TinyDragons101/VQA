import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from conversation import get_conv_template
from typing import List, Optional, Tuple, Union
from transformers import (AutoModel, GenerationConfig,)
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch.nn as nn
import json
import re
    
class CustomQwenVLCaptionModel:
    def __init__(
        self,
        # model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        model_name="Qwen/Qwen3-VL-8B-Instruct",
        device="cuda:0"
    ):
        self.device = torch.device(device)

        # Load model
        # self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     model_name,
        #     torch_dtype=torch.bfloat16,
        #     # load_in_4bit=True,
        #     device_map=device,
        #     trust_remote_code=True
        # ).eval()
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="flash_attention_2"
        ).eval()

        # 2. Processor tự động nhận diện template của Qwen3
        self.processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True,
            min_pixels=256 * 28 * 28,
            max_pixels=640 * 28 * 28
        )
        
        self.processor.tokenizer.padding_side = "left"

        print(f"✅ {model_name} loaded successfully on {device}")
        
    def extract_json_array(self, text):
        try:
            # Làm sạch text trước khi parse
            text = text.strip()
            if "```json" in text:
                text = text.split("```json")[-1].split("```")[0].strip()
            
            return json.loads(text)
        except:
            matches = re.findall(r'\[.*?\]', text, re.DOTALL)
            for m in matches:
                try:
                    return json.loads(m)
                except:
                    continue
        return None
    
    def extract_json_object(self, text):
        if not text:
            return None
        
        # Loại bỏ các block markdown thừa
        clean_text = re.sub(r'```json\s*|```', '', text).strip()
        
        # Tìm vị trí thực tế của JSON
        start_idx = clean_text.find('{')
        end_idx = clean_text.rfind('}')
        
        # Trường hợp model quên dấu { ở đầu (do ta mồi prompt)
        if start_idx == -1 and '"category"' in clean_text:
            clean_text = "{" + clean_text
            start_idx = 0
            if end_idx == -1: # Nếu cũng thiếu dấu đóng
                clean_text = clean_text + '}'
                end_idx = len(clean_text) - 1
        
        if start_idx == -1 or end_idx == -1:
            return None

        json_str = clean_text[start_idx : end_idx + 1]
        
        try:
            # Xóa các ký tự điều khiển gây lỗi parse (newline trong string, etc.)
            json_str = "".join(c for c in json_str if ord(c) >= 32 or c in "\n\r\t")
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Cố gắng cứu vớt nếu thiếu dấu đóng ngoặc kép do bị cắt ngang (truncation)
            try:
                return json.loads(json_str + '"}')
            except:
                return None

    @torch.no_grad()
    def generate_caption_and_category(
        self,
        image_path,
        prompt="Mô tả chi tiết hình ảnh văn hóa nghệ thuật này bằng tiếng Việt.",
        max_new_tokens=300,
        do_sample=True
    ):
        image = Image.open(image_path).convert("RGB")

        # Qwen3 tối ưu tốt hơn với system prompt rõ ràng về style
        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là một biên tập viên nội dung chuyên nghiệp, chuyên phân tích văn hóa và nghệ thuật Việt Nam. "
                    "Hãy mô tả hình ảnh một cách sâu sắc, giàu tính nghệ thuật và phân loại hoàn toàn bằng tiếng Việt dựa trên nội dung dưới đây."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Chuẩn bị input theo chuẩn chat template mới
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Xử lý ảnh: Qwen3 hỗ trợ dynamic resolution tốt hơn
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.device, dtype=torch.bfloat16)

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=0.7, 
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

        output_ids = self.model.generate(
            **inputs,
            generation_config=gen_config
        )

        # Tách phần trả về từ output
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        return self.extract_json_array(response)
    
    @torch.no_grad()
    def generate_batch(
        self,
        image_paths,
        prompts,
        max_new_tokens=300,
        do_sample=True
    ):
        """
        Xử lý song song một nhóm ảnh và prompts trên GPU.
        image_paths: List[str]
        prompts: List[str]
        """
        all_messages = []
        all_images = []

        # 1. Chuẩn bị Messages cho từng item trong batch
        for img_path, p in zip(image_paths, prompts):
            image = Image.open(img_path).convert("RGB")
            all_images.append(image)
            
            messages = [
                {
                    "role": "system",
                    "content": "Bạn là một biên tập viên nội dung chuyên nghiệp, chuyên phân tích văn hóa và nghệ thuật Việt Nam."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": p},
                    ],
                }
            ]
            # Apply chat template cho từng item
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            all_messages.append(text)

        # 2. Processor xử lý Batch (Quan trọng: padding=True)
        # Qwen3-VL processor sẽ tự động handle việc resize và padding ảnh/text
        inputs = self.processor(
            text=all_messages,
            images=all_images,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        # 3. Cấu hình generation cho Batch
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=0.7, 
            top_p=0.9,
            repetition_penalty=1.1,
            # Đảm bảo dùng đúng pad_token_id để tránh lỗi lệch batch
            pad_token_id=self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,
        )

        # 4. Model Generate (Matrix multiplication trên A100)
        output_ids = self.model.generate(
            **inputs,
            generation_config=gen_config
        )

        # 5. Decode kết quả cho cả batch
        # Cắt bỏ phần input prompt trong output_ids
        generated_ids = [
            out[len(ins):] for ins, out in zip(inputs.input_ids, output_ids)
        ]
        
        responses = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # 6. Parse JSON cho từng kết quả trong list
        final_results = []
        for res_text in responses:
            parsed = self.extract_json_object(res_text)            
            if parsed:
                final_results.append(parsed)
            else:
                # Fallback: cố gắng cứu vớt dữ liệu nếu parse thất bại
                print(f"Failed to parse, raw response: {res_text[:100]}...")
                final_results.append({
                    "category": "Khác", 
                    "caption": res_text.replace("{", "").replace("}", "").strip()[:300]
                })
        return final_results
    
    @torch.no_grad()
    def generate_questions(
        self,
        image_path,
        prompt="Dựa trên hình ảnh, hãy tạo 5 câu hỏi về các chi tiết nghệ thuật và văn hóa.",
        max_new_tokens=400
    ):
        
        image = load_and_resize_image(image_path, 512)
        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là một chuyên gia nghiên cứu văn hóa, lịch sử, nghệ thuật và kiến trúc Việt Nam."
                    "Dựa trên hình ảnh và thông tin bài báo, hãy đặt 5 câu hỏi VQA (Visual Question Answering) chuyên sâu."
                    "CHỈ trả về một JSON Array (ví dụ: [\"câu 1\", \"câu 2\"]). "
                    "Không có văn bản thừa ngoài JSON."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        ).to(self.device)

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False, # Tắt sampling để cấu trúc JSON ổn định
            repetition_penalty=1.1,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

        output_ids = self.model.generate(
            **inputs,
            generation_config=gen_config
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()

        # Parse JSON bằng hàm đã cải tiến
        questions = self.extract_json_array(response)
        
        if questions and isinstance(questions, list):
            return questions[:5]
        
        # Fallback nếu model trả về text thường
        return [line.strip("- ") for line in response.split('\n') if '?' in line][:5]
    
    @torch.no_grad()
    def generate_questions_batch(
        self,
        image_paths,
        prompts,
        max_new_tokens=500,
        do_sample=False
    ):
        """
        Tạo câu hỏi VQA theo lô (Batch) cho nhiều ảnh cùng lúc.
        """
        all_messages = []
        all_images = []

        # 1. Chuẩn bị dữ liệu đầu vào cho từng item trong batch
        for img_path, p in zip(image_paths, prompts):
            try:
                image = Image.open(img_path).convert("RGB")
                all_images.append(image)
                
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Bạn là một chuyên gia nghiên cứu văn hóa, lịch sử, nghệ thuật và kiến trúc Việt Nam. "
                            "Dựa trên hình ảnh và thông tin bài báo, hãy đặt 5 câu hỏi VQA (Visual Question Answering) chuyên sâu. "
                            "CHỈ trả về một JSON Array (ví dụ: [\"câu 1\", \"câu 2\"]). Không có văn bản thừa ngoài JSON."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": p},
                        ],
                    }
                ]
                
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                all_messages.append(text)
            except Exception as e:
                print(f" lỗi load ảnh {img_path}: {e}")
                # Thêm dummy data để giữ đúng index của batch nếu 1 ảnh lỗi
                all_messages.append("") 
                all_images.append(Image.new('RGB', (224, 224), color='white'))

        # 2. Tokenize và chuẩn bị Tensor
        inputs = self.processor(
            text=all_messages,
            images=all_images,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        # 3. Cấu hình Generation
        # Với câu hỏi (JSON), tắt do_sample giúp output ổn định hơn
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=0.2 if do_sample else None,
            repetition_penalty=1.1,
            pad_token_id=self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,
        )

        # 4. Model Inference
        output_ids = self.model.generate(
            **inputs,
            generation_config=gen_config
        )

        # 5. Decode kết quả
        generated_ids = [
            out[len(ins):] for ins, out in zip(inputs.input_ids, output_ids)
        ]
        
        responses = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # 6. Parse kết quả và xử lý Fallback
        batch_questions = []
        for res_text in responses:
            res_text = res_text.strip()
            
            # Ưu tiên parse JSON Array
            questions = self.extract_json_array(res_text)
            
            # Fallback nếu parse JSON thất bại
            if not questions or not isinstance(questions, list):
                # Tách dòng và lấy các dòng có dấu chấm hỏi
                questions = [
                    line.strip("- ").strip() 
                    for line in res_text.split('\n') 
                    if '?' in line
                ]
            
            # Đảm bảo trả về tối đa 5 câu và không bị rỗng
            final_qs = questions[:5] if questions else ["Bạn thấy gì trong bức ảnh này?"]
            batch_questions.append(final_qs)

        return batch_questions
    
    @torch.no_grad()
    def generate_answers(self, image_path, prompt):
        image = load_and_resize_image(image_path, 512)
        messages = [
            {
                "role": "system",
                "content": "Bạn là chuyên gia văn hóa. Trả lời danh sách câu hỏi dưới dạng JSON Array. Không giải thích thêm."
            },
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)

        output_ids = self.model.generate(
            **inputs,
            generation_config=GenerationConfig(
                max_new_tokens=1024,
                do_sample=False,
                temperature=0.1,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        )

        response = self.processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )[0].strip()

        # Sử dụng lại hàm extract_json_array có sẵn của bạn
        return self.extract_json_array(response)
    
    @torch.no_grad()
    def generate_answers_batch(self, image_paths, prompts):
        """
        Đã sửa đổi: Nhận trực tiếp list image_paths và prompts 
        để khớp với file pipeline của bạn.
        """
        all_messages = []
        all_images = []

        for img_path, p in zip(image_paths, prompts):
            try:
                image = Image.open(img_path).convert("RGB")
                all_images.append(image)
                
                messages = [
                    {
                        "role": "system",
                        "content": "Bạn là chuyên gia nghiên cứu văn hóa, nghệ thuật. Trả lời dựa trên bài báo và hình ảnh. TRẢ VỀ DUY NHẤT một JSON Array theo định dạng: [[\"câu hỏi 1\", \"câu trả lời 1\"], [...]]"
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": p},
                        ],
                    }
                ]
                
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                all_messages.append(text)
            except Exception as e:
                print(f"[-] Lỗi load ảnh {img_path}: {e}")
                all_messages.append("") 
                all_images.append(Image.new('RGB', (224, 224), color='white'))

        # Xử lý Batch qua Processor
        inputs = self.processor(
            text=all_messages,
            images=all_images,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        # Cấu hình generation
        gen_config = GenerationConfig(
            max_new_tokens=1024,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,
        )

        # Inference
        output_ids = self.model.generate(**inputs, generation_config=gen_config)

        # Decode kết quả
        generated_ids = [out[len(ins):] for ins, out in zip(inputs.input_ids, output_ids)]
        responses = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Parse JSON từng kết quả trong batch
        final_batch_results = []
        for res_text in responses:
            parsed = self.extract_json_array(res_text)
            final_batch_results.append(parsed)

        return final_batch_results

    @torch.no_grad()
    def generate_difficulty_batch(self, image_paths, prompts):
        """
        Đánh giá độ khó của các cặp QA theo lô (Batch).
        Trả về list các JSON array chứa mức độ khó ["1", "3", ...].
        """
        all_messages = []
        all_images = []

        for img_path, p in zip(image_paths, prompts):
            try:
                image = Image.open(img_path).convert("RGB")
                all_images.append(image)
                
                messages = [
                    {
                        "role": "system",
                        "content": "Bạn là chuyên gia đánh giá dữ liệu VQA. Hãy đánh giá độ khó của các cặp câu hỏi - câu trả lời dựa trên bài báo và hình ảnh. TRẢ VỀ DUY NHẤT một JSON Array chứa các mức độ khó từ \"1\" đến \"5\"."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": p},
                        ],
                    }
                ]
                
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                all_messages.append(text)
            except Exception as e:
                print(f"[-] Lỗi load ảnh {img_path}: {e}")
                all_messages.append("") 
                all_images.append(Image.new('RGB', (224, 224), color='white'))

        # Xử lý Batch qua Processor
        inputs = self.processor(
            text=all_messages,
            images=all_images,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        # Cấu hình generation
        gen_config = GenerationConfig(
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,
        )

        # Inference
        output_ids = self.model.generate(**inputs, generation_config=gen_config)

        # Decode kết quả
        generated_ids = [out[len(ins):] for ins, out in zip(inputs.input_ids, output_ids)]
        responses = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Parse JSON từng kết quả trong batch
        final_batch_results = []
        for res_text in responses:
            parsed = self.extract_json_array(res_text)
            final_batch_results.append(parsed)

        return final_batch_results
    
def load_and_resize_image(path, max_size=384):
    image = Image.open(path).convert("RGB")
    
    # Resize giữ aspect ratio
    image.thumbnail((max_size, max_size))
    
    return image