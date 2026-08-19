from terminus.config import CONFIG
from terminus.observability.logging import get_logger

logger = get_logger(__name__)


def get_llm():
    """Initialises and returns the LLM client."""
    
    provider = CONFIG["llm"]["provider"]
    model = CONFIG["llm"]["model"]
    logger.info(f"Using LLM Provider: {provider}, Model: {model}")
    
    if provider.lower() == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model)
    elif provider.lower() == "cerebras":
        from langchain_cerebras import ChatCerebras
        return ChatCerebras(model=model)
    elif provider.lower() == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model)
    else:
        raise ValueError(f"LLM Provider not found: {provider}")

def get_embedder():
    """Initialises and returns the embedder client."""
    provider = CONFIG["embeddings"]["provider"]
    model = CONFIG["embeddings"]["model"]
    logger.info(f"Using Embedder Provider: {provider}, Model: {model}")
    
    if provider.lower() == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model)
    elif provider.lower() == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model)
    else:
        raise ValueError(f"Embedder Provider not found: {provider}")