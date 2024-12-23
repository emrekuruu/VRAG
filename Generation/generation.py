from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from PIL import Image
from io import BytesIO
import base64

class QAOutput(BaseModel):
    reasoning: str
    answer: str

class Faithfulness(BaseModel):
    reasoning: str 
    score: str

IMAGE_BASED_FAITHFULNESS_PROMPT = PromptTemplate(
    input_variables=["query", "schema", "answer"],
    template="""
    You are an expert evaluator tasked with assessing the faithfulness of a response provided by an AI model based solely on the PDF pages you receive. Faithfulness means the response is accurate and supported by the provided content.

    Instructions:
    - You will receive a question, the AI's response, and PDF pages containing the context.
    - Analyze the response sentence by sentence and evaluate how well it aligns with the context.
    - Provide a detailed reasoning chain and assign a faithfulness score between 0 and 1.
      - `1.0`: Completely faithful and accurate
      - `0.0`: Entirely unsupported or contradictory

    Produce your output in this strict JSON format:
    {schema}

    Question: {query}
    Answer: {answer}
    """
)

TEXT_BASED_FAITHFULNESS_PROMPT = PromptTemplate(
    input_variables=["query", "context", "schema", "answer"],
    template="""
    You are an expert evaluator tasked with assessing the faithfulness of a response provided by an AI model based solely on the textual context you receive. Faithfulness means the response is accurate and supported by the provided content.

    Instructions:
    - You will receive a question, the AI's response, and the textual context.
    - Analyze the response sentence by sentence and evaluate how well it aligns with the context.
    - Provide a detailed reasoning chain and assign a faithfulness score between 0 and 1.
      - `1.0`: Completely faithful and accurate
      - `0.0`: Entirely unsupported or contradictory

    Produce your output in this strict JSON format:
    {schema}

    Context: {context}
    Question: {query}
    Answer: {answer}
    """
)

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
    - The 'answer' field should contain only the final concise answer, with no extra reasoning steps.

    Be sure to consider all PDF pages provided. Base your reasoning and final answer solely on the content of these pages.

    Question: {query}
    """
)


TEXT_PROMPT = PromptTemplate(
    input_variables=["query", "context", "schema"],
    template="""
    You are a highly-skilled financial analyst with expertise in reviewing and analyzing financial information.
    You will receive some textual context (financial data, narrative, or supporting information) and you must answer 
    the user's question based solely on that context.

    You must produce your answer in the following strict JSON format:
    {schema}

    Instructions:
    - The 'reasoning' field should contain a chain-of-thought as a numbered list, detailing how you leverage the given 
      textual context to arrive at your answer:
      For example:
      1 - Extract key financial metrics or data points
      2 - Apply relevant financial analysis or logical reasoning
      3 - Synthesize findings to answer the question
    - The 'answer' field should contain only the final concise answer, with no reasoning steps included.

    Base your reasoning and final answer solely on the provided context below.

    Context: {context}

    Question: {query}
    """
)

async def image_based(query, pages):
    llm = ChatOpenAI(model="gpt-4o")
    llm = llm.with_structured_output(QAOutput)
    schema = QAOutput.model_json_schema()

    prompt = IMAGE_PROMPT.format(query=query, schema=schema)

    # Construct the message content with PDF pages as images
    content = [
        {"type": "text", "text": prompt},
    ]
    for p in pages:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/pdf;base64,{p}"}
        })

    message = HumanMessage(content=content)
    response: QAOutput = await llm.ainvoke([message])
    response = {"reasoning": response.reasoning, "answer": response.answer}
    return response
    
async def evaluate_faithfulness(query, answer, context, type="text"):

    llm = ChatOpenAI(model="gpt-4o")
    llm = llm.with_structured_output(Faithfulness)
    schema = Faithfulness.model_json_schema()

    if type == "image":
        prompt = IMAGE_BASED_FAITHFULNESS_PROMPT.format(query=query, schema=schema, answer=answer)

        content = [
            {"type": "text", "text": prompt},
        ]
        for p in context: 
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{p}"}
            })

        message = HumanMessage(content=content)
        response: Faithfulness = await llm.ainvoke([message])

    elif type == "text":
        prompt = TEXT_BASED_FAITHFULNESS_PROMPT.format(query=query, context=context, schema=schema, answer=answer)
        response: Faithfulness = await llm.ainvoke(prompt)

    return {"reasoning": response.reasoning, "score": response.score}

async def text_based(query, chunks):
    llm = ChatOpenAI(model="gpt-4o")
    llm = llm.with_structured_output(QAOutput)
    schema = QAOutput.model_json_schema()

    prompt = TEXT_PROMPT.format(query=query, context=chunks, schema=schema)
    response: QAOutput = await llm.ainvoke(prompt)
    response = {"reasoning": response.reasoning, "answer": response.answer}
    return response
