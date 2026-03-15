import os
import json
import time
import concurrent.futures
from typing import List, Dict, Any
from google import genai
from google.genai import types
from google.oauth2 import credentials
from PIL import Image
from dotenv import load_dotenv
import tqdm

load_dotenv()

class GeminiVLCaptionModel:
    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gemini-2.0-flash",
    ):
        """
        Khởi tạo Gemini API Client sử dụng SDK google-genai mới.
        """
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION")
        access_token = os.getenv("GCP_ACCESS_TOKEN")

        if not all([project_id, location, access_token]):
            raise ValueError("❌ Thiếu cấu hình GCP trong file .env (Project, Location hoặc Token)")
        
        creds = credentials.Credentials(access_token)
        
        # Khởi tạo Client mới
        self.client = genai.Client(
            vertexai=True, 
            project=project_id, 
            location=location,
            credentials=creds
        )
        self.model_name = model_name
        
        print(f"✅ Gemini SDK (google-genai) loaded: {model_name}")

    def generate_questions(
        self,
        image_path: str,
        prompt: str = "Dựa trên hình ảnh, hãy tạo 7 câu hỏi về các chi tiết nghệ thuật và văn hóa."
    ) -> List[str]:
        """
        Tạo 5 câu hỏi VQA dưới dạng JSON Array sử dụng SDK mới.
        """
        try:
            # Mở ảnh bằng PIL
            image = Image.open(image_path)
            
            # Cấu hình hệ thống và format đầu ra
            config = types.GenerateContentConfig(
                system_instruction="Bạn là chuyên gia nghiên cứu văn hóa Việt Nam. Hãy trả về kết quả duy nhất dưới dạng một JSON Array chứa các chuỗi văn bản (string). Không giải thích gì thêm.",
                temperature=0.4,
                response_mime_type="application/json",
            )

            # Gọi API (Thứ tự trong SDK mới là nội dung trước, config sau)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image],
                config=config
            )

            # Xử lý text trả về
            if not response.text:
                return []

            # Parse JSON an toàn
            data = json.loads(response.text.strip())
            
            # Xử lý trường hợp model trả về object lồng list
            if isinstance(data, dict):
                for val in data.values():
                    if isinstance(val, list):
                        return val[:8]
            
            return data[:8] if isinstance(data, list) else []

        except Exception as e:
            # Xử lý lỗi Rate Limit (429) hoặc lỗi kết nối
            if "429" in str(e):
                print(f"[!] Rate limit hit for {image_path}, waiting...")
                time.sleep(5)
            else:
                print(f"[-] Error processing {image_path}: {e}")
            return []

    def generate_questions_batch(self, image_paths, prompts, max_workers=2):
        """
        Xử lý song song bằng ThreadPoolExecutor.
        """
        results = [None] * len(image_paths)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.generate_questions, path, p): i 
                for i, (path, p) in enumerate(zip(image_paths, prompts))
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results
    
    def generate_answers(self, image_path: str, prompt: str) -> List[str]:
        """
        Gửi ảnh và danh sách câu hỏi qua Gemini, yêu cầu trả về list câu trả lời JSON.
        """
        try:
            image = Image.open(image_path)
            
            # Chỉ thị hệ thống ép model trả về đúng định dạng JSON array
            config = types.GenerateContentConfig(
                system_instruction=(
                    "Bạn là chuyên gia phân tích hình ảnh và báo chí. "
                    "Dựa trên nội dung bài báo và hình ảnh, hãy trả lời các câu hỏi được cung cấp. "
                    "TRẢ VỀ DUY NHẤT một JSON Array chứa các chuỗi văn bản (string) tương ứng với thứ tự câu hỏi. "
                    "Ví dụ: [\"Câu trả lời 1\", \"Câu trả lời 2\"]. Không giải thích gì thêm."
                ),
                temperature=0.2, # Giảm nhiệt độ để câu trả lời chính xác hơn
                response_mime_type="application/json",
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image],
                config=config
            )

            if not response.text:
                return []

            data = json.loads(response.text.strip())
            
            # Xử lý nếu model trả về dict lồng list
            if isinstance(data, dict):
                for val in data.values():
                    if isinstance(val, list): return val
            return data if isinstance(data, list) else []

        except Exception as e:
            if "429" in str(e):
                time.sleep(10) # Chờ lâu hơn nếu bị giới hạn quota
            print(f"[-] Error tại {os.path.basename(image_path)}: {e}")
            return []

    def generate_answers_parallel(self, tasks, max_workers=5):
        """
        Chạy song song các request API.
        """
        results = [None] * len(tasks)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.generate_answers, t['image_path'], t['prompt']): i 
                for i, t in enumerate(tasks)
            }
            for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(tasks), desc="Gemini API Calling"):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results