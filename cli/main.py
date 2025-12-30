"""CLI main entry point.

Stock Prediction ML Pipeline CLI.
Commands: train, predict, evaluate, monitor
"""

import click

from .train import train
from .predict import predict
from .evaluate import evaluate
from .monitor import monitor
from .pipeline import pipeline


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """🚀 Stock Prediction ML Pipeline CLI
    
    Professional command-line interface for ML operations.
    
    Examples:
        stock-ml train --ticker PETR4.SA
        stock-ml predict --model-path artifacts/models/best_model.pt --ticker PETR4.SA
        stock-ml evaluate --model-path artifacts/models/best_model.pt --ticker PETR4.SA
        stock-ml pipeline --mode both --ticker PETR4.SA
        stock-ml monitor
    """
    pass


# Register commands
cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)
cli.add_command(monitor)
cli.add_command(pipeline)


if __name__ == "__main__":
    cli()
