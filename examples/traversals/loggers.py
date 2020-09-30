import logging

from rich.console import Console
from rich.logging import RichHandler


# logging.basicConfig(
#     level="DEBUG",
#     format="%(message)s",
#     datefmt="[%X]",
#     handlers=[RichHandler(markup=True, show_time=False)]
# )

console = Console()
logger = logging.getLogger("examples.traversals")
logger.addHandler(RichHandler(console=console, markup=True, show_time=False))
