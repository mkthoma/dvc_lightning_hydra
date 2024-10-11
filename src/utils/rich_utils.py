from pathlib import Path
from typing import Sequence, Any, Dict

import rich
import rich.syntax
import rich.tree
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.console import Console
from rich.panel import Panel

from utils.pylogger import get_pylogger

log = get_pylogger(__name__)

console = Console()


def print_config_tree(
    config: Dict[str, Any],
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints content of config using Rich library and its tree structure."""

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        (
            queue.append(field)
            if field in config
            else log.warning(
                f"Field '{field}' not found in config. Skipping '{field}' config printing..."
            )
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in config:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = config[field]
        if isinstance(config_group, dict):
            branch_content = str(config_group)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    console.print(tree)

    # save config tree to file
    if save_to_file:
        # Ensure the 'logs' directory exists
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Now write to the log file
        log_file_path = logs_dir / "config_tree.log"
        with open(log_file_path, "w") as file:
            file.write("Your log content here...\n")  # Adjust this part as needed

def print_rich_progress(text: str):
    """Prints a rich progress bar with spinner."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task(description=text, total=100)
        while not progress.finished:
            progress.update(task, advance=0.9)


def print_rich_panel(text: str, title: str):
    """Prints a rich panel with given text and title."""
    console.print(Panel(text, title=title))
