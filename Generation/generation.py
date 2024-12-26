
from . import Openai, Claude, Google

async def image_based(query, pages, model_type):
    if model_type == "openai":
        return await Openai.image_based(query, pages)
    elif model_type == "claude":
        return await Claude.image_based(query, pages)
    elif model_type == "google":
        return await Google.image_based(query, pages)
    
async def text_based(query, chunks, model_type):
    if model_type == "openai":
        return await Openai.text_based(query, chunks)
    elif model_type == "claude":
        return await Claude.text_based(query, chunks)
    elif model_type == "google":
        return await Google.text_based(query, chunks)

async def hybrid(query, pages, chunks, model_type):
    if model_type == "openai":
        return await Openai.hybrid(query, pages, chunks)
    elif model_type == "claude":
        return await Claude.hybrid(query, pages, chunks)
    elif model_type == "google":
        return await Google.hybrid(query, pages, chunks)
