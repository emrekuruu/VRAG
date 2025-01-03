from pydantic import BaseModel
from langchain.prompts import PromptTemplate
import os 
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import asyncio 
import re 
import json

with open(f".keys/openai_api_key.txt", "r") as file:
    os.environ["OPENAI_API_KEY"] = file.read().strip()
    api_key = file.read().strip()
class QAOutput(BaseModel):
    reasoning: str
    answer: str

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

JSONIFY_PROMPT = PromptTemplate(
    input_variables=["response_string", "schema"],
    template="""
    As an AI JSON Parser, analyze the provided response string and extract the JSON object according to the following specifications:

    1. Input Format:
    - Look for JSON content enclosed within triple backticks (```)
    - Ignore any preliminary text before the JSON object
    - Maintain exact content without alterations

    2. Required Schema Fields:
    - reasoning: Extract the numbered list detailing the step-by-step thought process
    - answer: Extract the concise final answer

    Schema to match:
    {schema}

    3. Output Requirements:
    - Return a valid JSON object matching the provided schema
    - Preserve all original content exactly as presented
    - Ensure proper JSON formatting and structure

    Parse the following response string and return the extracted JSON object:

    {response_string}
    """
)

async def jsonify(response_string):
    llm = ChatOpenAI(model="gpt-4o", api_key=api_key) 
    llm = llm.with_structured_output(QAOutput)
    schema = QAOutput.model_json_schema()

    prompt = JSONIFY_PROMPT.format(response_string=response_string, schema=schema)
    response: QAOutput = await exponential_backoff(llm.ainvoke, prompt)
    response = {"reasoning": response.reasoning, "answer": response.answer}
    return response


async def get_structured_output(response):
  
    try:
      response = {"reasoning": response.reasoning, "answer": response.answer}
      return response
    
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
            return response
        
        except:
            return await jsonify(response_string=response)


IMAGE_PROMPT = PromptTemplate(
    input_variables=["query", "schema"],
    template="""
    You are a highly-skilled financial analyst with deep expertise in evaluating financial documents and extracting insights.
    You will receive PDF pages containing relevant information. Your task is to answer the user's financial-related question 
    based solely on these PDF pages.

    You must produce your answer in the following strict JSON format:
    {schema}

    Instructions:
    - The 'reasoning' field should contain your detailed chain-of-thought as a numbered list, showing how you derive the answer:
      For example:
      1 - Identify relevant financial figures or metrics from the pages
      2 - Apply financial reasoning or calculations
      3 - Cross-check figures or context
      4 - Arrive at the final conclusion

    - The 'answer' field should be extremely concise. Your response should be in the form of numerical values, percentages, or an extremely short and simple sentence, including units where relevant. 
      For example: '0.5%', '$500.0', '2 billion gallons', or 'B-'.

    Be sure to consider all PDF pages provided. Base your reasoning and final answer solely on the content of these pages.

    Question: {query}
    """
)

TEXT_PROMPT = PromptTemplate(
    input_variables=["query", "context", "schema"],
    template="""
     You are a highly-skilled financial analyst with deep expertise in evaluating financial documents and extracting insights.
    You will receive some textual context (financial data, narrative, or supporting information). Your task is to answer the user's financial-related question 
    based solely on this context.

    You must produce your answer in the following strict JSON format:
    
    {schema}

    Instructions:
    - The 'reasoning' field should contain a chain-of-thought as a numbered list, detailing how you leverage the given 
      textual context to arrive at your answer:
      For example:
      1 - Identify relevant financial information or metrics from the context
      2 - Apply financial reasoning or calculations
      3 - Cross-check context
      4 - Arrive at the final conclusion

    - The 'answer' field should be extremely concise. Your response should be in the form of numerical values, percentages, or an extremely short and simple sentence, including units where relevant. 
      For example: '0.5%', '$500.0', '2 billion gallons', or 'B-'.

    Base your reasoning and final answer solely on the provided context below.

    Context: {context}

    Question: {query}
    """
)

HYBRID_PROMPT = PromptTemplate(
    input_variables=["query", "context", "schema"],
    template="""
    You are a highly-skilled financial analyst with deep expertise in evaluating financial documents and extracting insights.
    You will receive multi-modal context in the form of both PDF pages (image-based) and textual content. Your task is to answer the user's financial-related question 
    based solely on this combined input. Examine the given textual context first, then while you are examining the image context, give more attention to the sections of the images that intersect with the textual contexts.
    Start by reviewing the textual context, then focus on the image context, especially on the sections of the images that intersect with the textual content. However, the final answer should prioritize information from the images.

    You must produce your answer in the following strict JSON format:
    {schema}

    Instructions:
    - The 'reasoning' field should contain a chain-of-thought as a numbered list, detailing how you leverage the given 
      textual context to arrive at your answer:
      For example:
      1 - Identify relevant financial information or metrics from the textual context
      2 - Check for any intersections between the textual and image contexts
      3 - Identify relevant financial figures or metrics from the pdf pages giving more attention to the sections that intersect with the textual context
      4 - Apply financial reasoning or calculations
      5 - Cross-check figures or context
      6 - Arrive at the final conclusion

    - The 'answer' field should be extremely concise. Your response should be in the form of numerical values, percentages, or an extremely short and simple sentence, including units where relevant. 
      For example: '0.5%', '$500.0', '2 billion gallons', or 'B-'.

    Base your reasoning and final answer solely on the provided textual and image context.

    Textual Context: {context}

    Question: {query}
    """
)
