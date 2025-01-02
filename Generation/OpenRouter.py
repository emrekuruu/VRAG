from openai import OpenAI
from .prompts import IMAGE_PROMPT, TEXT_PROMPT, HYBRID_PROMPT,  QAOutput, jsonify
import asyncio
import re 
import json 

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

    response = await query_model_async(messages, model, structured=structured)

    try:
        response = {"reasoning": response.reasoning, "answer": response.answer}
    except:
        response = response.content
        json_pattern = r'```json\n({.*?})\n```'
        matches = re.findall(json_pattern, ''.join(response), re.DOTALL)
        try:
            reasoning_and_answer_json = matches[-1] 
            parsed_response = json.loads(reasoning_and_answer_json)
            reasoning = parsed_response["reasoning"]
            answer = parsed_response["answer"]

            if type(answer) != str:
                raise Exception("Answer must be a string")
            
            response = {"reasoning": reasoning, "answer": answer}
        except:
            return await jsonify(response_string=response)
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

    response = await query_model_async(messages, model, structured=structured)

    try:
        response = {"reasoning": response.reasoning, "answer": response.answer}
    except:
        json_pattern = r'```json\n({.*?})\n```'
        matches = re.findall(json_pattern, ''.join(response), re.DOTALL)
        try:
            reasoning_and_answer_json = matches[-1] 
            parsed_response = json.loads(reasoning_and_answer_json)
            reasoning = parsed_response["reasoning"]
            answer = parsed_response["answer"]

            if type(answer) != str:
                raise Exception("Answer must be a string")
            
            response = {"reasoning": reasoning, "answer": answer}
        except:
            return await jsonify(response_string=response)
        
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

    response = await query_model_async(messages, model, structured=structured)
    response = {"reasoning": response.reasoning, "answer": response.answer} if structured else response
    return response
