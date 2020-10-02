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


class WarnOnce:
    def __init__(self):
        self.warnings = set()

    def warn_once(self, msg):
        if msg not in self.warnings:
            logger.warning(msg)
            self.warnings.add(msg.strip())

    def __call__(self, msg):
        return self.warn_once(msg)


warn_once = WarnOnce()
