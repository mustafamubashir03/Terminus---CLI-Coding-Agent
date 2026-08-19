from pathlib import Path

import chromadb
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt

from terminus.config import CONFIG
from terminus.context.indexers.semantic_chroma import index_codebase
from terminus.llm.factory import get_embedder, get_llm
from terminus.observability.logging import get_logger

load_dotenv()
console = Console()
logger = get_logger(__name__)

def initialize():
    logger.info("Initializing Terminus...")
    llm = get_llm()
    embedder = get_embedder()
    collection = get_or_create_index()
    logger.info("Terminus initialized successfully")
    return llm, embedder, collection


def run():
    logger.info("Starting Terminus CLI")
    console.print("[bold blue]Welcome to Terminus![/bold blue]")
    console.print("Type [bold red]'/exit'[/bold red] or [bold red]'/quit'[/bold red] to quit")
    console.print("Type [bold cyan]'/clear'[/bold cyan] to clear the screen")
    _llm, _embedder, collection = initialize()
    while True:
        query = Prompt.ask("[bold green]Query[/bold green]")
        user_input = query.lower()
        logger.info(f"Query: {query}")
        console.print(f"Response: {query}")
        if user_input == "":
            console.print("[bold red]Please enter a query[/bold red]")
            continue
        elif user_input in ["/exit","/quit"]:
            console.print("[bold blue]Goodbye![/bold blue]")
            break
        elif user_input == "/clear":
            console.clear()
            continue
        elif user_input.startswith("/ask"):
            question = query.removeprefix("/ask ").strip()
            if not question:
                console.print("[bold red]Please enter a question[/bold red]")
                continue
            console.print(f"[bold green]Question:[/bold green] {question}")
        elif user_input.startswith("/show_semantic_index"):
            console.print("[bold green]Showing semantic index...[/bold green]")
            show_semantic_index(collection)
            
        else:
            logger.warning("Invalid query", extra={"query": query})
            console.print("[bold red]Invalid query[/bold red]")
            console.print("[yellow] Unknown command Try :  /ask 'Question Here' /clear")
            console.print("Use '/ask <question>' to ask a question about codebase")
            console.print("[bold yellow]show_semantic_index[/bold yellow] for showing semantic index")
        
        
    

    
def get_or_create_index()->chromadb.Collection:
    chroma_client = chromadb.PersistentClient(path=CONFIG["chromadb"]["persist_dir"])
    collection = chroma_client.get_or_create_collection(
        name=CONFIG["chromadb"]["collection_name"],
        embedding_function=get_embedder()
    )
    if collection.count() > 0:
        logger.info(f"Found existing index with {collection.count()} chunks.")
        console.print(f"[dim]Loading existing index-{collection.count()} chunks [/dim]")
        return collection
    repo_path = str(Path.cwd())
    logger.info(f"No existing index found. Initializing new index for {repo_path}")
    console.print(f"[yellow]No index found. Initializing new index for [/yellow]{repo_path}")
    return index_codebase(repo_path)

def show_semantic_index(collection:chromadb.Collection)->None:
    """Show the semantic index stats"""
    console.print("[bold green]Semantic Index Stats:[/bold green]")
    console.print(f"[dim]Collection:[/dim] {collection.name}")
    console.print(f"[dim]Total Chunks:[/dim] {collection.count()}")
    results = collection.get(include=['documents','metadatas',"embeddings"])
    for i,(doc, meta, emb) in enumerate(zip(results['documents'],results['metadatas'],results['embeddings'])):
        console.print(f"Chunk {i+1}:\n")
        console.print(f"[bold green]Metadata:[/bold green] {meta}")
        console.print(f"[bold green]Content:[/bold green] {doc}")
        console.print(f"[bold green]Embedding:[/bold green] {emb[:20]}...")
        console.print("-"*50)
        