from langchain_core.messages import HumanMessage
import asyncio
from .prompts import IMAGE_PROMPT, TEXT_PROMPT, HYBRID_PROMPT, QAOutput
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define a wrapper for Qwen 2.5B
class ChatQwenGenerativeAI:
    def __init__(self, model, api_key=None):
        self.model_name = model
        self.api_key = api_key  # Not used for local inference
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, device_map="auto")

    def with_structured_output(self, output_parser):
        self.output_parser = output_parser
        return self

    async def ainvoke(self, messages):
        prompt = messages[0].content[0]["text"]
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
        output_ids = self.model.generate(input_ids, max_length=512, temperature=0.7)

        response_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Parse the response into the structured output
        return self.output_parser.parse({"response": response_text})

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
    llm = ChatQwenGenerativeAI(model="Qwen/Qwen2.5-72B-Instruct") 
    llm = llm.with_structured_output(QAOutput)
    schema = QAOutput.model_json_schema()

    prompt = IMAGE_PROMPT.format(query=query, schema=schema)

    content = [{"type": "text", "text": prompt}]
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
    llm = ChatQwenGenerativeAI(model="Qwen/Qwen2.5-72B-Instruct") 
    llm = llm.with_structured_output(QAOutput)
    schema = QAOutput.model_json_schema()

    prompt = TEXT_PROMPT.format(query=query, context=chunks, schema=schema)
    response: QAOutput = await exponential_backoff(llm.ainvoke, [prompt])
    response = {"reasoning": response.reasoning, "answer": response.answer}
    return response

async def hybrid(query, pages, chunks):
    llm = ChatQwenGenerativeAI(model="Qwen/Qwen2.5-72B-Instruct")  
    llm = llm.with_structured_output(QAOutput)
    schema = QAOutput.model_json_schema()

    prompt = HYBRID_PROMPT.format(query=query, context=chunks, schema=schema)

    content = [{"type": "text", "text": prompt}]
    for p in pages:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{p}"}
        })

    message = HumanMessage(content=content)
    response: QAOutput = await exponential_backoff(llm.ainvoke, [message])
    response = {"reasoning": response.reasoning, "answer": response.answer}
    return response
