from langchain_core.messages import HumanMessage
import asyncio
from .prompts import IMAGE_PROMPT, TEXT_PROMPT, HYBRID_PROMPT, QAOutput
from langchain_anthropic import ChatAnthropic

with open(f"/Users/emrekuru/Developer/VRAG/.keys/claude_api_key.txt",  "r") as file:
    claude = file.read().strip()

async def exponential_backoff(func, *args, retries=100, initial_wait=60, **kwargs):
    wait_time = initial_wait
    for attempt in range(retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            print(e)
            if attempt == retries - 1:
                raise e
            await asyncio.sleep(wait_time)
            wait_time += 10

async def image_based(query, pages):
    llm = ChatAnthropic(api_key=claude, model="claude-3-5-sonnet-20241022") 
    llm = llm.with_structured_output(QAOutput)
    schema = QAOutput.model_json_schema()

    prompt = IMAGE_PROMPT.format(query=query, schema=schema)

    content = [
        {"type": "text", "text": prompt},
    ]
    for p in pages:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg", 
                "data": p[0]
            }
        })

    message = HumanMessage(content=content)
    response: QAOutput = await exponential_backoff(llm.ainvoke, [message])
    response = {"reasoning": response.reasoning, "answer": response.answer}
    return response
    
async def text_based(query, chunks):
    llm =  ChatAnthropic(api_key=claude, model="claude-3-5-sonnet-20241022") 
    llm = llm.with_structured_output(QAOutput)
    schema = QAOutput.model_json_schema()

    prompt = TEXT_PROMPT.format(query=query, context=chunks, schema=schema)
    response: QAOutput = await exponential_backoff(llm.ainvoke, prompt)
    response = {"reasoning": response.reasoning, "answer": response.answer}
    return response

async def hybrid(query, pages, chunks):
    llm =  ChatAnthropic(api_key=claude, model="claude-3-5-sonnet-20241022") 
    llm = llm.with_structured_output(QAOutput)
    schema = QAOutput.model_json_schema()

    prompt = HYBRID_PROMPT.format(query=query, context=chunks, schema=schema)

    content = [
        {"type": "text", "text": prompt},
    ]

    for p in pages:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg", 
                "data": p[0]
            }
        })

    message = HumanMessage(content=content)
    response: QAOutput = await exponential_backoff(llm.ainvoke, [message])
    response = {"reasoning": response.reasoning, "answer": response.answer}
    return response