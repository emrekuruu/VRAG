from openai import OpenAI
from .prompts import IMAGE_PROMPT, TEXT_PROMPT, HYBRID_PROMPT, QAOutput, exponential_backoff, get_structured_output
import asyncio

with open(".keys/deep_api_key.txt", "r") as file:
    api_key = file.read().strip()

client = OpenAI(
    base_url="https://api.deepinfra.com/v1/openai",
    api_key=api_key
)

schema = QAOutput.model_json_schema()
structured = True

async def query_model_async(messages, model, structured = True):

    def sync_query_model():
    
        if structured:
            try:
                completion = client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=QAOutput,
                )
                return completion.choices[0].message.parsed
        
            except Exception as e:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
        else:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
            )
        
        return completion.choices[0].message.content
        
    return await asyncio.to_thread(sync_query_model)

    
async def image_based(query, pages, model):
    global structured
    
    prompt = IMAGE_PROMPT.format(query=query, schema=schema)

    if model.lower() == "qwen/qvq-72b-preview":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ] 
            }
        ]
        structured = False

    else:

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

    response = await exponential_backoff(query_model_async, messages, model)
    response = await get_structured_output(response)
    return response

async def text_based(query, chunks, model):
    global structured

    prompt = TEXT_PROMPT.format(query=query, context=chunks, schema=schema)

    if model.lower() == "qwen/qvq-72b-preview":
        messages = [
            {
                "role": "user",
                "content": prompt
                
            }
        ]
        structured = False

    else:

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

    response = await exponential_backoff(query_model_async, messages, model)
    response = await get_structured_output(response)
    return response

async def hybrid(query, pages, chunks, model):

    prompt = HYBRID_PROMPT.format(query=query, context=chunks, schema=schema)
    
    if model == "qwen/qvq-72b-preview":
        messages = [
            {
                "role": "user",
                "content": prompt
                
            }
        ]
        structured = False
    
    else:  
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

    response = await exponential_backoff(query_model_async, messages, model)
    response = await get_structured_output(response)
    return response
