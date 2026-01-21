"""Train command - Full training pipeline using refactored TrainPipeline."""

import sys
from pathlib import Path

import click
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.pipeline.train_pipeline import TrainPipeline

# B3 ticker categories (43 unique tickers total)
B3_TICKERS = {
    "blue_chips": [
        "PETR4.SA",  # Petrobras
        "VALE3.SA",  # Vale
        "ITUB4.SA",  # Itaú
        "BBDC4.SA",  # Bradesco
        "ABEV3.SA",  # Ambev
        "BBAS3.SA",  # Banco do Brasil
        "WEGE3.SA",  # WEG
        "RENT3.SA",  # Localiza
        "B3SA3.SA",  # B3
        "SUZB3.SA",  # Suzano
    ],
    "bancos": [
        "SANB11.SA",  # Santander
        "BBSE3.SA",  # BB Seguridade
    ],
    "energia": [
        "PETR3.SA",  # Petrobras PN
        "ELET3.SA",  # Eletrobras
        "ELET6.SA",  # Eletrobras PNB
        "CMIG4.SA",  # Cemig
        "CPLE6.SA",  # Copel
    ],
    "varejo": [
        "MGLU3.SA",  # Magazine Luiza
        "LREN3.SA",  # Lojas Renner
        "PETZ3.SA",  # Petz
        "AMER3.SA",  # Americanas
    ],
    "mineracao": [
        "CMIN3.SA",  # CSN Mineração
        "GOAU4.SA",  # Metalúrgica Gerdau
    ],
    "construcao": [
        "CYRE3.SA",  # Cyrela
        "BEEF3.SA",  # Minerva
        "EZTC3.SA",  # EZTec
    ],
    "telecom": [
        "VIVT3.SA",  # Vivo
        "TIMS3.SA",  # Tim
    ],
    "papel_celulose": [
        "KLBN11.SA",  # Klabin
    ],
    "saude": [
        "RADL3.SA",  # Raia Drogasil
        "HAPV3.SA",  # Hapvida
        "FLRY3.SA",  # Fleury
    ],
    "tecnologia": [
        "TOTS3.SA",  # Totvs
        "LWSA3.SA",  # Locaweb
    ],
    "alimentacao": [],
    "servicos": [
        "CSAN3.SA",  # Cosan
        "RAIL3.SA",  # Rumo
    ],
}

# All tickers (all categories combined - 43 unique tickers)
ALL_TICKERS = sorted(
    list(
        set(
            ticker
            for category_tickers in B3_TICKERS.values()
            for ticker in category_tickers
        )
    )
)


@click.command()
@click.option(
    "--ticker", type=str, default=None, help="Single stock ticker (e.g., PETR4.SA)"
)
@click.option(
    "--tickers", type=str, default=None, help="Multiple tickers separated by comma"
)
@click.option(
    "--category",
    type=click.Choice(
        [
            "blue_chips",
            "bancos",
            "energia",
            "varejo",
            "mineracao",
            "construcao",
            "telecom",
            "papel_celulose",
            "saude",
            "tecnologia",
            "alimentacao",
            "servicos",
        ],
        case_sensitive=False,
    ),
    default=None,
    help="Ticker category",
)
@click.option(
    "--use-all-tickers",
    is_flag=True,
    default=False,
    help=f"Use ALL {len(ALL_TICKERS)} available tickers",
)
@click.option(
    "--start-date", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)"
)
@click.option("--hidden-size", type=int, default=100, help="LSTM hidden size")
@click.option("--num-layers", type=int, default=3, help="Number of LSTM layers")
@click.option("--dropout", type=float, default=0.3, help="Dropout rate")
@click.option("--lr", type=float, default=0.001, help="Learning rate")
@click.option("--epochs", type=int, default=100, help="Maximum training epochs")
@click.option("--batch-size", type=int, default=64, help="Batch size")
@click.option(
    "--experiment-name", type=str, default=None, help="MLflow experiment name"
)
@click.option(
    "--model-path",
    type=str,
    default="artifacts/models/best_model.pt",
    help="Model save path",
)
@click.option("--seed", type=int, default=42, help="Random seed")
def train(
    ticker,
    tickers,
    category,
    use_all_tickers,
    start_date,
    hidden_size,
    num_layers,
    dropout,
    lr,
    epochs,
    batch_size,
    experiment_name,
    model_path,
    seed,
):
    """🚂 Train LSTM model (single or multi-ticker) using MLflow-first architecture.

    Examples:
        # Single ticker
        stock-predict train --ticker PETR4.SA --epochs 50

        # Multi-ticker (category)
        stock-predict train --category blue_chips --epochs 50

        # Multi-ticker (custom)
        stock-predict train --tickers PETR4.SA,VALE3.SA --epochs 50

        # All tickers
        stock-predict train --use-all-tickers --epochs 50
    """
    # Determine ticker list
    if use_all_tickers:
        ticker_list = ALL_TICKERS
    elif category:
        ticker_list = B3_TICKERS[category]
    elif tickers:
        ticker_list = [t.strip() for t in tickers.split(",")]
    elif ticker:
        ticker_list = None  # Single-ticker mode
        single_ticker = ticker
    else:
        raise click.UsageError(
            "Must provide --ticker, --tickers, --category, or --use-all-tickers"
        )

    try:
        if ticker_list:
            # Multi-ticker
            pipeline = TrainPipeline(
                tickers=ticker_list,
                start_date=start_date,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                learning_rate=lr,
                epochs=epochs,
                batch_size=batch_size,
                experiment_name=experiment_name,
                model_save_path=model_path,
                seed=seed,
            )
        else:
            # Single-ticker
            pipeline = TrainPipeline(
                ticker=single_ticker,
                start_date=start_date,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                learning_rate=lr,
                epochs=epochs,
                batch_size=batch_size,
                experiment_name=experiment_name,
                model_save_path=model_path,
                seed=seed,
            )

        # Run training
        results = pipeline.run()

        logger.success("\n" + "=" * 60)
        logger.success("✅ Training Complete!")
        logger.success("=" * 60)
        logger.success(f"Model saved: {model_path}")
        if experiment_name:
            logger.success(f"Experiment: {experiment_name}")
        logger.success("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise click.ClickException(str(e))
