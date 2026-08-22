
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt
from terminus.context.indexers.factory import get_or_create_indexer,show_index
from terminus.llm.factory import get_embedder, get_llm
from terminus.agent.orchestrator import handle_query
from terminus.observability.logging import get_logger

load_dotenv()
console = Console()
logger = get_logger(__name__)

def initialize():
    logger.info("Initializing Terminus...")
    llm = get_llm()
    repo_path = str(Path.cwd())
    embedder = get_embedder()
    index = get_or_create_indexer(repo_path)
    logger.info("Terminus initialized successfully")
    return llm, embedder, index


def run():
    logger.info("Starting Terminus CLI")
    console.print("[bold blue]Welcome to Terminus![/bold blue]")
    console.print("Type [bold red]'/exit'[/bold red] or [bold red]'/quit'[/bold red] to quit")
    console.print("Type [bold cyan]'/clear'[/bold cyan] to clear the screen")
    _llm, _embedder, index = initialize()
    while True:
        query = Prompt.ask("[bold green]Query[/bold green]")
        user_input = query.lower()
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
            response = handle_query(question)
            console.print(f"[bold blue]Response:[/bold blue] {response}")
        elif user_input.startswith("/show_semantic_index"):
            console.print("[bold green]Showing semantic index...[/bold green]")
            show_index(index)
        elif user_input.startswith("/help"):
            console.print("[bold green]Help:[/bold green]")
            console.print("\n[bold green]Commands:[/bold green]")
            console.print("[yellow] /ask 'Question Here' - Ask a question about codebase[/yellow]")
            console.print("[yellow] /clear - Clear the screen[/yellow]")
            console.print("[yellow] /exit - Exit the CLI[/yellow]")
            console.print("[yellow] /quit - Exit the CLI[/yellow]")
            console.print("[yellow] /help - Show this help message[/yellow]")
            console.print("[yellow] /show_semantic_index - Show semantic index stats[/yellow]")
        else:
            logger.warning("Invalid query", extra={"query": query})
            console.print("[bold red]Invalid query[/bold red]")
            console.print("[yellow] Unknown command Try :  /ask 'Question Here' /clear")
            console.print("Use '/ask <question>' to ask a question about codebase")
            console.print("[bold yellow]show_semantic_index[/bold yellow] for showing semantic index")
        


        