from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt

load_dotenv()
from terminus.observability.logging import get_logger

console = Console()
logger = get_logger(__name__)

def run():
    logger.info("Starting Terminus CLI")
    console.print("[bold blue]Welcome to Terminus![/bold blue]")
    console.print("Type [bold red]'exit'[/bold red] to quit")
    while True:
        query = Prompt.ask("[bold green]Query[/bold green]")
        logger.info(f"Query: {query}")
        console.print(f"Response: {query}")
        if query.strip() == "":
            console.print("[bold red]Please enter a query[/bold red]")
            continue
        if query.lower() in ["/exit","/quit"]:
            console.print("[bold blue]Goodbye![/bold blue]")
            break
        if query.lower() == "/clear":
            console.clear()
            continue
        else:
            console.print(f"[bold red]Response:[/bold red] {query}")
            
        
    

    
    