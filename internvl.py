import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
from transformers import AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from conversation import get_conv_template
from typing import List, Optional, Tuple, Union
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch.nn as nn
import json
import re


class CustomInternVLCaptionModel14B():
    def __init__(self, model_name = "OpenGVLab/InternVL-14B-224px" , device='cuda:7'):
        
        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            device_map=None
        ).to(self.device).eval()

        self.image_processor = CLIPImageProcessor.from_pretrained(
            model_name, trust_remote_code=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, add_eos_token=True, trust_remote_code=True)
        self.tokenizer.pad_token_id = 0  

    def encode_image(self, images, mode='InternVL-G', is_path = False):
        if is_path:
            images = [Image.open(path).convert('RGB') for path in images]

        pixel_values = self.image_processor(images=images, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
        embedding = self.model.encode_image(pixel_values, mode=mode)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding
    
    def encode_text(self, text):
        prefix = 'summarize:'
        text = prefix + text 
        input_ids = self.tokenizer([text], return_tensors='pt', max_length=80,
                      truncation=True, padding='max_length').input_ids.to(self.device)
        feature_text = self.model.encode_text(input_ids)
        feature_text = feature_text / feature_text.norm(dim=-1, keepdim=True)
        return feature_text

    def compute_image_text_probs(self, image, text, mode='InternVL-G', is_image_path = False, soft_max = True):
        with torch.no_grad():
            image = self.encode_image(image, mode=mode, is_path=is_image_path)
            text = self.encode_text(text)
            image = image / image.norm(dim=-1, keepdim=True)
            text = text / text.norm(dim=-1, keepdim=True)
            logit_scale = self.model.logit_scale.exp()
            probs = logit_scale * image @ text.t()
            if soft_max:
                probs = probs.softmax(dim=-1)
            else:
                probs = probs
            return probs
    
    
    def compute_text_text_probs(self, text1, text2, soft_max = True):
        with torch.no_grad():
            text_feature_1 = self.encode_text(text1)
            text_feature_2 = self.encode_text(text2)
            text_feature_1 = text_feature_1 / text_feature_1.norm(dim=-1, keepdim=True)
            text_feature_2 = text_feature_2 / text_feature_2.norm(dim=-1, keepdim=True)
            logit_scale = self.model.logit_scale.exp()
            probs = logit_scale * text_feature_1 @ text_feature_2.t()
            if soft_max:
                probs = probs.softmax(dim=-1)
            else:
                probs = probs
            return probs
        
    
    def crop_center(self, image, crop_width, crop_height):
        width, height = image.size
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height
        return image.crop((left, top, right, bottom))

    def generate_caption(self, image, is_path = False):
        with torch.no_grad():
            self.tokenizer.add_eos_token = False
            if is_path:
                image = Image.open(image).convert('RGB')
            
            
            pixel_values = self.image_processor(images=image, return_tensors='pt').pixel_values
            pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
            tokenized = self.tokenizer("English caption:", return_tensors='pt')
            pred = self.model.generate(
                pixel_values=pixel_values,
                input_ids= tokenized.input_ids.to(self.device),
                attention_mask= tokenized.attention_mask.to(self.device),
                num_beams=5,
                max_new_tokens=32  # required
            )
            caption = self.tokenizer.decode(pred[0].cpu(), skip_special_tokens=True).strip()    
            return caption
        

class CustonInternVLCaptionModel():
    def __init__(self, model_name = 'OpenGVLab/InternVL2_5-8B' , device='cuda:0'):
        self.device = torch.device(device)
        
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval().to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

        self.text_projection = nn.Parameter(torch.empty(4096, 1024)).to(device)  # frozen

    def build_transform(self, input_size, aug=False):
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD

        if aug:
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ])
        else:
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ])
        return transform


        
    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images
    
    def load_image(self, image_file, input_size=448, max_num=12, aug=False):
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size, aug=aug)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    def pure_text_generation(self, question):
        with torch.no_grad():
            generation_config = dict(max_new_tokens=256, do_sample=True)
            response, history = self.model.chat(self.tokenizer, None, question, generation_config, history=None, return_history=True)
            return response
        
    
    def generate_caption(self, image_path):
        with torch.no_grad():
            image = self.load_image(image_path, max_num=12).to(torch.bfloat16).to(self.device)
            generation_config = dict(max_new_tokens=256, do_sample=True)


            question = '<image>\nPlease describe detailed the image in a paragraph'
            response, history = self.model.chat(self.tokenizer, image, question, generation_config, history=None, return_history=True)
            return response
        
    def generate__short_caption(self, image_path):
        with torch.no_grad():
            image = self.load_image(image_path, max_num=12).to(torch.bfloat16).to(self.device)
            generation_config = dict(max_new_tokens=256, do_sample=True)


            question = '<image>\nPlease describe the unique features espescially the posting human or the features of that image of the image in one or two sentence'
            response, history = self.model.chat(self.tokenizer, image, question, generation_config, history=None, return_history=True)
            return response
    
    
    def generate_captions(self, image_paths):
        num_patches_list = []
        final_pixels = None  # <- initialize here

        for image_path in image_paths:
            
            pixels = self.load_image(image_path, max_num=12).to(torch.bfloat16).to(self.device)
            num_patches_list.append(pixels.size(0))
            final_pixels = torch.cat((final_pixels, pixels), dim=0) if final_pixels is not None else pixels

        questions = ['<image>\nPlease describe the image in a very paragraph'] * len(num_patches_list)
        generation_config = dict(max_new_tokens=256, do_sample=True)
        responses = self.model.batch_chat(
            self.tokenizer,
            final_pixels,
            num_patches_list=num_patches_list,
            questions=questions,
            generation_config= generation_config
        )
        return responses

    
    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
            num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
            verbose=False):
        
        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        template = get_conv_template(self.model.template)
        template.system_message = self.model.system_message
        
        stop_str = None
        if template.sep2 and template.sep2.strip():
            stop_str = template.sep2.strip()
        elif template.sep and template.sep.strip():
            stop_str = template.sep.strip()
        else:
            stop_str = tokenizer.eos_token

        eos_token_id = tokenizer.convert_tokens_to_ids(stop_str)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')
        

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
        
        

        model_inputs = tokenizer(query, return_tensors='pt')
        
        device = torch.device(self.model.language_model.device if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        
        
        attention_mask = model_inputs['attention_mask'].to(device)
        generation_config['eos_token_id'] = eos_token_id
        
        gen_config = GenerationConfig(
            max_new_tokens=generation_config.get("max_new_tokens", 200),
            do_sample=generation_config.get("do_sample", False),
            temperature=generation_config.get("temperature", 0.7),
            repetition_penalty=generation_config.get("repetition_penalty", 1.2),
            eos_token_id=eos_token_id,
        )
        
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config,
            # **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        
        sep_to_split = template.sep.strip() if (template.sep and template.sep.strip()) else None
        if sep_to_split:
            response = response.split(sep_to_split)[0].strip()
        else:
            sep2_to_split = template.sep2.strip() if (template.sep2 and template.sep2.strip()) else None
            if sep2_to_split:
                response = response.split(sep2_to_split)[0].strip()
            else:
                response = response.strip()
                
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.model.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.model.extract_feature(pixel_values)
            
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.model.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)

        outputs = self.model.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
    
    def get_inputs_embeddings(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ):
        assert self.model.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.model.extract_feature(pixel_values)
            
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.model.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        
        return input_embeds
    

    
    def get_embedding(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
            num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
            verbose=False):
        
        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        template = get_conv_template(self.model.template)
        template.system_message = self.model.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
        
        

        model_inputs = tokenizer(query, return_tensors='pt')
        
        device = torch.device(self.model.language_model.device if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        
        
        attention_mask = model_inputs['attention_mask'].to(device)
        generation_config['eos_token_id'] = eos_token_id
        embeddings = self.get_inputs_embeddings(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        return embeddings, attention_mask
    
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
            local_files_only=True
        ).eval()

        # 2. Processor tự động nhận diện template của Qwen3
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

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
        ).to(self.device)

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
    def generate_questions(
        self,
        image_path,
        prompt="Dựa trên hình ảnh, hãy tạo 5 câu hỏi về các chi tiết nghệ thuật và văn hóa.",
        max_new_tokens=400
    ):
        image = Image.open(image_path).convert("RGB")

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