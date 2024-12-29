from openai import OpenAI
import json  
from .prompts import OPEN_ROUTER_IMAGE_PROMPT, OPEN_ROUTER_HYBRID_PROMPT, OPEN_ROUTER_TEXT_PROMPT
import asyncio

with open(".keys/openrouter_api_key.txt", "r") as file:
    api_key = file.read().strip()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

async def query_model_async(messages, model):

    def sync_query_model():
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error querying model: {e}")
            raise e
        
    return await asyncio.to_thread(sync_query_model)

def extract_and_load_json(raw_string):
    try:
        start = raw_string.find("{")
        end = raw_string.rfind("}")

        if start == -1 or end == -1 or start > end:
            raise ValueError("No valid JSON object found in the string.")

        json_string = raw_string[start:end + 1]

        return json.loads(json_string)
    except Exception as e:
        print(raw_string)
        print(f"Error extracting and loading JSON: {e}")
        return None
    
async def image_based(query, pages, model):

    prompt = OPEN_ROUTER_IMAGE_PROMPT.format(query=query)

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]

    for p in pages:
        messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{p}"}
        })

    response_content = await query_model_async(messages, model)
    response_dict = extract_and_load_json(response_content)
    return response_dict

async def text_based(query, chunks, model):
    prompt = OPEN_ROUTER_TEXT_PROMPT.format(query=query, context=chunks)
    messages = [{"role": "system", "content": prompt}]
    response_content = await query_model_async(messages, model)
    response_dict = extract_and_load_json(response_content)
    return response_dict

async def hybrid(query, pages, chunks, model):

    prompt = OPEN_ROUTER_HYBRID_PROMPT.format(query=query, context=chunks)

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]

    for p in pages:
        messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{p}"}
        })

    response_content = await query_model_async(messages, model)
    response_dict = extract_and_load_json(response_content)
    return response_dict
