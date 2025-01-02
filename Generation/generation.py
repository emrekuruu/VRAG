
from . import OpenRouter, Openai, Claude, Google

async def image_based(query, pages, model_type):
    provider = model_type.split("-")[0]
    model = model_type[len(provider)+1:]

    if provider == "openai":
        return await Openai.image_based(query, pages,  model)
    elif provider == "claude":
        return await Claude.image_based(query, pages,  model)
    elif provider == "google":
        return await Google.image_based(query, pages,  model)
    elif provider == "openrouter":
        return await OpenRouter.image_based(query, pages, model)

    
async def text_based(query, chunks, model_type):
    provider = model_type.split("-")[0]
    model = model_type[len(provider)+1:]

    if provider == "openai":
        return await Openai.text_based(query, chunks,  model)
    elif provider == "claude":
        return await Claude.text_based(query, chunks,  model)
    elif provider == "google":
        return await Google.text_based(query, chunks, model) 
    elif provider == "openrouter":
        return await OpenRouter.text_based(query, chunks,  model)


async def hybrid(query, pages, chunks, model_type):
    provider = model_type.split("-")[0]
    model = model_type[len(provider)+1:]

    if provider == "openai":
        return await Openai.hybrid(query, pages, chunks,  model)
    elif provider == "claude":
        return await Claude.hybrid(query, pages, chunks,  model)
    elif provider == "google":
        return await Google.hybrid(query, pages, chunks,  model)
    elif provider == "openrouter":
        return await OpenRouter.hybrid(query, pages, chunks,  model)
