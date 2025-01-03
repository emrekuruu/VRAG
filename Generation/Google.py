from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from .prompts import IMAGE_PROMPT, TEXT_PROMPT, HYBRID_PROMPT, QAOutput, exponential_backoff, get_structured_output
from langchain_community.callbacks import get_openai_callback

with open(f"/Users/emrekuru/Developer/VRAG/.keys/google_api_key.txt",  "r") as file:
    google = file.read().strip()

async def image_based(query, pages, model_type):
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

    response = await get_structured_output(response.content)
    return response

async def text_based(query, chunks, model_type):
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

    response = await get_structured_output(response.content)
    return response

async def hybrid(query, pages, chunks, model_type):
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

    response = await get_structured_output(response.content)
    return response