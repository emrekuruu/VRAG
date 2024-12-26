from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import asyncio
from .prompts import IMAGE_PROMPT, TEXT_PROMPT, HYBRID_PROMPT, QAOutput

async def exponential_backoff(func, *args, retries=20, initial_wait=10, **kwargs):
    wait_time = initial_wait
    for attempt in range(retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt == retries - 1:
                raise e
            await asyncio.sleep(wait_time)
            wait_time += 10

async def image_based(query, pages):
    llm = ChatOpenAI(model="gpt-4o") 
    llm = llm.with_structured_output(QAOutput)
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
    response: QAOutput = await exponential_backoff(llm.ainvoke, [message])
    response = {"reasoning": response.reasoning, "answer": response.answer}
    return response
    
async def text_based(query, chunks):
    llm = ChatOpenAI(model="gpt-4o")
    llm = llm.with_structured_output(QAOutput)
    schema = QAOutput.model_json_schema()

    prompt = TEXT_PROMPT.format(query=query, context=chunks, schema=schema)
    response: QAOutput = await exponential_backoff(llm.ainvoke, prompt)
    response = {"reasoning": response.reasoning, "answer": response.answer}
    return response

async def hybrid(query, pages, chunks):
    llm = ChatOpenAI(model="gpt-4o") 
    llm = llm.with_structured_output(QAOutput)
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
    response: QAOutput = await llm.ainvoke([message])
    response = {"reasoning": response.reasoning, "answer": response.answer}
    return response