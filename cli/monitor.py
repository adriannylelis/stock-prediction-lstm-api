"""Monitor command - Launch MLflow UI."""

import click
import subprocess
from loguru import logger


@click.command()
@click.option(
    '--port',
    type=int,
    default=5000,
    help='Port for MLflow UI'
)
@click.option(
    '--host',
    type=str,
    default='127.0.0.1',
    help='Host for MLflow UI'
)
@click.option(
    '--backend-store-uri',
    type=str,
    default='file:./mlruns',
    help='MLflow tracking URI'
)
def monitor(port: int, host: str, backend_store_uri: str):
    """📡 Launch MLflow UI for experiment monitoring.
    
    Opens web interface at http://localhost:5000
    
    Example:
        stock-ml monitor
        stock-ml monitor --port 8080
    """
    logger.info(f"🚀 Launching MLflow UI at http://{host}:{port}")
    logger.info(f"Backend: {backend_store_uri}")
    logger.info("Press Ctrl+C to stop")
    
    try:
        subprocess.run([
            'mlflow', 'ui',
            '--backend-store-uri', backend_store_uri,
            '--host', host,
            '--port', str(port)
        ])
    except KeyboardInterrupt:
        logger.info("\n👋 MLflow UI stopped")
    except Exception as e:
        logger.error(f"❌ Failed to start MLflow UI: {e}")
        raise click.ClickException(str(e))
