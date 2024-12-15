from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

IMAGE_PROMPT = PromptTemplate(
    input_variables=["query"],
    template="""
     Answer the following question based solely on the following PDF pages. Examine **all** of the given pages before you begin your answer. Provide a numbered reasoning for your answer, with clear logic steps leading to the conclusion.

     Example:

     1- Step one reasoning...

     2- Step two reasoning...

     Final answer: [your answer]

    Question: {query}
    """
)

TEXT_PROMPT = PromptTemplate(
    input_variables=["query", "context"],
    template="""
    Answer the following question based solely on the following context. Provide a numbered reasoning for your answer, with clear logic steps leading to the conclusion.

    Example:

    1- Step one reasoning...

    2- Step two reasoning...

    Final answer: [your answer]

    Context: {context}

    Question: {query}
    """
)

async def image_based(query, pages):
    llm = ChatOpenAI(model="gpt-4o")

    prompt = IMAGE_PROMPT.format(query=query)

    content = [
        {"type": "text", "text": prompt},
    ]

    print(len(pages))

    for p in pages:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/pdf;base64,{p}"}
        })

    print(len(content))

    message = HumanMessage(content=content)     
    response = await llm.ainvoke([message])
    return response.content

async def text_based(query, chunks):
    llm = ChatOpenAI(model="gpt-4o")

    prompt = TEXT_PROMPT.format(query=query, context=chunks)

    response = await llm.ainvoke(prompt)
    return response.content
