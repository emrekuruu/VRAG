import json
import asyncio
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from .prompts import OPEN_ROUTER_IMAGE_PROMPT, OPEN_ROUTER_HYBRID_PROMPT, OPEN_ROUTER_TEXT_PROMPT
import torch

torch.cuda.empty_cache()
torch.cuda.memory_summary(device="cuda")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/QVQ-72B-Preview", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/QVQ-72B-Preview")

# Function to extract and load JSON
def extract_and_load_json(raw_string):
    try:
        start = raw_string.find("{")
        end = raw_string.rfind("}")
        if start == -1 or end == -1 or start > end:
            raise ValueError("No valid JSON object found in the string.")
        json_string = raw_string[start:end + 1]
        return json.loads(json_string)
    except Exception as e:
        print(f"Error extracting and loading JSON: {e}")
        return None

# Async function to query the Qwen model
async def query_qwen_async(messages):
    def sync_query_qwen():
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=8192)
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
    
    return await asyncio.to_thread(sync_query_qwen)

# Image-based query
async def image_based(query, pages):

    prompt = OPEN_ROUTER_IMAGE_PROMPT.format(query=query)

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }]

    for p in pages:
        messages[0]["content"].append({
            "type": "image",
            "image": f"data:image/base64,{p}"
        })

    response_content = await query_qwen_async(messages)
    print(response_content)
    response_dict = extract_and_load_json(response_content)
    return response_dict

# Text-based query
async def text_based(query, context):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": f"Analyze the following text: {context}"}]
        }
    ]
    response_content = await query_qwen_async(messages)
    response_dict = extract_and_load_json(response_content)
    return response_dict

# Hybrid query
async def hybrid(query, pages, context):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": f"Analyze the following: {query}. Context: {context}"}]
        }
    ]
    for p in pages:
        messages[0]["content"].append({
            "type": "image",
            "image": {"url": f"data:image/jpeg;base64,{p}"}
        })

    response_content = await query_qwen_async(messages)
    response_dict = extract_and_load_json(response_content)
    return response_dict
