from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
from .prompts import IMAGE_PROMPT, TEXT_PROMPT, HYBRID_PROMPT, QAOutput, jsonify
from langchain_community.callbacks import get_openai_callback
import re 
import json

with open(f"/Users/emrekuru/Developer/VRAG/.keys/google_api_key.txt",  "r") as file:
    google = file.read().strip()

async def exponential_backoff(func, *args, retries=20, initial_wait=10, **kwargs):
    wait_time = initial_wait
    for attempt in range(retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt == retries - 1:
                raise e
            print(e)
            await asyncio.sleep(wait_time)
            wait_time += 10

async def image_based(query, pages, model_type="gemini-2.0-flash-exp"):
    llm = ChatGoogleGenerativeAI(model=model_type, api_key=google)  
    llm = llm.with_structured_output(QAOutput) if "thinking" not in model_type else llm
    schema = QAOutput.model_json_schema()

    prompt = IMAGE_PROMPT.format(query=query, schema=schema)

    content = [
        {"type": "text", "text": prompt},
    ]
    for p in pages:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{p}"}
        })

    message = HumanMessage(content=content)

    with get_openai_callback() as cb:
        response: QAOutput = await exponential_backoff(llm.ainvoke, [message])
        print(f"Total Tokens: {cb.total_tokens}")

    try:
        response = {"reasoning": response.reasoning, "answer": response.answer}
    except:
        response = response.content
        json_pattern = r'```json\n({.*?})\n```'
        matches = re.findall(json_pattern, ''.join(response), re.DOTALL)
        if matches:
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

async def text_based(query, chunks, model_type="gemini-2.0-flash-exp"):
    llm = ChatGoogleGenerativeAI(model=model_type, api_key=google)  
    llm = llm.with_structured_output(QAOutput) if "thinking" not in model_type else llm
    schema = QAOutput.model_json_schema()

    prompt = TEXT_PROMPT.format(query=query, context=chunks, schema=schema)

    content = [
        {"type": "text", "text": prompt},
    ]
    message = HumanMessage(content=content)

    with get_openai_callback() as cb:
        response: QAOutput = await exponential_backoff(llm.ainvoke, [message])
        print(f"Total Tokens: {cb.total_tokens}")

    try:
        response = {"reasoning": response.reasoning, "answer": response.answer}
    except:
        response = response.content
        json_pattern = r'```json\n({.*?})\n```'
        matches = re.findall(json_pattern, ''.join(response), re.DOTALL)
        if matches:
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

async def hybrid(query, pages, chunks, model_type="gemini-2.0-flash-exp"):
    llm = ChatGoogleGenerativeAI(model=model_type, api_key=google)    
    llm = llm.with_structured_output(QAOutput) if "thinking" not in model_type else llm
    schema = QAOutput.model_json_schema()

    prompt = HYBRID_PROMPT.format(query=query, context=chunks, schema=schema)

    content = [
        {"type": "text", "text": prompt},
    ]
    
    for p in pages:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{p}"}
        })

    message = HumanMessage(content=content)

    with get_openai_callback() as cb:
        response: QAOutput = await exponential_backoff(llm.ainvoke, [message])
        print(f"Total Tokens: {cb.total_tokens}")

    try:
        response = {"reasoning": response.reasoning, "answer": response.answer}
    except:
        response = response.content
        json_pattern = r'```json\n({.*?})\n```'
        matches = re.findall(json_pattern, ''.join(response), re.DOTALL)
        if matches:
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