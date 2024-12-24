import base64
from PIL import Image
from io import BytesIO
import tempfile
import os
from ragas.metrics import MultiModalFaithfulness
from ragas.dataset_schema import SingleTurnSample
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

def decode_base64_images_to_paths(base64_images):
    temp_paths = []
    for base64_image in base64_images:
        image_data = base64.b64decode(base64_image[0])
        image = Image.open(BytesIO(image_data))
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        image.save(temp_file.name, format="PNG")
        temp_paths.append(temp_file.name)
    return temp_paths

async def evaluate_multimodal_faithfulness(query, answer, context):
    image_paths = decode_base64_images_to_paths(context)

    llm = ChatOpenAI(model="gpt-4o")

    try:
        sample = SingleTurnSample(
            user_input=query,
            response=answer,
            retrieved_contexts=image_paths
        )

        scorer = MultiModalFaithfulness()
        scorer.llm = LangchainLLMWrapper(llm)
        score = await scorer.single_turn_ascore(sample)

    finally:
        for path in image_paths:
            if os.path.exists(path):
                os.remove(path)

    return score
