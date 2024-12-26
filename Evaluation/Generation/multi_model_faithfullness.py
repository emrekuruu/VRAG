import base64
from PIL import Image
from io import BytesIO
import tempfile
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

class Faithfulness(BaseModel):
    reasoning: str 
    score: str

MULTIMODAL_FAITHFULNESS_PROMPT = PromptTemplate(
    input_variables=["query", "text_context", "answer"],
    template=""" 
    You are a multi-modal context faithfulness evaluation assistant. 
    Faithfulness" means that every factual statement in the response must be supported by the provided content. 
    The provided context can include text, images, or both. 

    **Instructions for Analysis:**

    1. Decompose the answer into individual sentences.

    2. For each sentence, assign one of the following labels based on the context:
    - **`supported`**: The sentence is explicitly supported (entailed) by the context. Provide an exact excerpt from the context that fully supports it.
    - **`contradictory`**: The sentence is falsified by the context. Provide an exact excerpt that shows the contradiction.
    - **`unsupported`**: The sentence is neither fully supported nor contradicted by the context. No excerpt is needed.
    - **`no_rad`**: The sentence does not require factual attribution (opinions, questions, greetings, etc.). No excerpt is needed.

    3. Provide a short rationale for each label, separate from the excerpt.

    4. Be strict with assigning `supported` and `contradictory`. If you cannot find direct evidence in the context, consider it `unsupported`.

    You must produce the final output as a sequence of JSON objects, one per line, each with the following fields:
    - `"sentence"`: The sentence under analysis.
    - `"label"`: One of `supported`, `unsupported`, `contradictory`, or `no_rad`.
    - `"rationale"`: Short explanation justifying the label.
    - `"excerpt"`: Relevant excerpt from the context for `supported` and `contradictory` only (otherwise `null`).

    **Input Provided:**
    - **Textual Context**: {context}
    - **Question**: {query}
    - **Answer**: {answer}

    **Task**: Output the sentence-by-sentence analysis in valid JSON lines following the schema described.

    Begin your analysis now.
    """
    )

# Currently not used but can be used if we use a metric that requires paths rather than base64 images
def decode_base64_images_to_paths(base64_images):
    temp_paths = []
    for base64_image in base64_images:
        image_data = base64.b64decode(base64_image[0])
        image = Image.open(BytesIO(image_data))
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        image.save(temp_file.name, format="PNG")
        temp_paths.append(temp_file.name)
    return temp_paths

async def evaluate_multimodel_faithfullness(query, answer, context, type="text"):
    llm = ChatOpenAI(model="gpt-4o")
    llm = llm.with_structured_output(Faithfulness)
    schema = Faithfulness.model_json_schema()

    if type == "image":

        text_context = "No textual context provided only images."
        prompt = MULTIMODAL_FAITHFULNESS_PROMPT.format(query=query, text_context=text_context, schema=schema, answer=answer)

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
        prompt = MULTIMODAL_FAITHFULNESS_PROMPT.format(query=query, text_context=context, schema=schema, answer=answer)
        response: Faithfulness = await llm.ainvoke(prompt)

    return {"reasoning": response.reasoning, "score": response.score}