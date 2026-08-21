from terminus.config import CONFIG
from terminus.observability.logging import get_logger

logger = get_logger(__name__)

_embedder_cache: dict = {}


def get_llm():
    """Initialises and returns the LLM client."""
    
    provider = CONFIG["llm"]["provider"]
    model = CONFIG["llm"]["model"]
    logger.info(f"Using LLM Provider: {provider}, Model: {model}")
    if provider.lower() == "fireworks":
        import os
        from langchain_fireworks import ChatFireworks
        api_key = os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY is not set in the .env file")
        return ChatFireworks(
            model=f"accounts/fireworks/models/{model}",
            temperature=0,
            timeout=None
        )
    
    elif provider.lower() == "openai":
        import os
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the .env file")
        return ChatOpenAI(model=model, api_key=api_key)
        
    elif provider.lower() == "cerebras":
        import os
        from langchain_cerebras import ChatCerebras
        api_key = os.environ.get("CEREBRAS_API_KEY")
        if not api_key:
            raise ValueError("CEREBRAS_API_KEY is not set in the .env file")
        return ChatCerebras(model=model, api_key=api_key)
        
    elif provider.lower() == "anthropic":
        import os
        from langchain_anthropic import ChatAnthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in the .env file")
        return ChatAnthropic(model=model, api_key=api_key)
        
    else:
        raise ValueError(f"LLM Provider not found: {provider}")

def get_embedder():
    """Initialises and returns the embedder client. Cached as a singleton to avoid reloading weights."""
    global _embedder_cache
    provider = CONFIG["embeddings"]["provider"]
    model = CONFIG["embeddings"]["model"]
    cache_key = f"{provider}:{model}"

    if cache_key in _embedder_cache:
        return _embedder_cache[cache_key]

    logger.info(f"Using Embedder Provider: {provider}, Model: {model}")

    if provider.lower() == "cerebras":
        from langchain_cerebras import CerebrasEmbeddings
        embedder = CerebrasEmbeddings(model=model)
    elif provider.lower() == "openai":
        from langchain_openai import OpenAIEmbeddings
        embedder = OpenAIEmbeddings(model=model)
    elif provider.lower() == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        embedder = HuggingFaceEmbeddings(model_name=model)
    else:
        raise ValueError(f"Embedder Provider not found: {provider}")

    _embedder_cache[cache_key] = embedder
    return embedder