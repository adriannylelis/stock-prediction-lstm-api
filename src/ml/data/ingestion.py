"""Stock data ingestion from Yahoo Finance.

This module handles downloading historical stock data from Yahoo Finance
using the yfinance library.
"""

from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger


class StockDataIngestion:
    """Handler for downloading stock data from Yahoo Finance.

    This class provides methods to download and validate stock price data
    from Yahoo Finance for B3 (Brazilian stocks) and international markets.

    Attributes:
        ticker: Stock ticker symbol (e.g., 'PETR4.SA' for Petrobras).
        start_date: Start date for data collection.
        end_date: End date for data collection.

    Example:
        >>> ingestion = StockDataIngestion(ticker="PETR4.SA", start_date="2020-01-01")
        >>> df = ingestion.download()
        >>> print(df.head())
    """

    def __init__(
        self, ticker: str, start_date: str | datetime, end_date: Optional[str | datetime] = None
    ) -> None:
        """Initialize the data ingestion handler.

        Args:
            ticker: Stock ticker symbol. For B3 stocks, use format: 'PETR4.SA'.
            start_date: Start date for data collection (YYYY-MM-DD or datetime).
            end_date: End date for data collection. If None, uses today.

        Raises:
            ValueError: If start_date is invalid or after end_date.
        """
        self.ticker = ticker.upper()

        # Parse dates
        self.start_date = self._parse_date(start_date)
        self.end_date = self._parse_date(end_date) if end_date else datetime.now()

        # Validate dates
        if self.start_date >= self.end_date:
            raise ValueError(
                f"start_date ({self.start_date}) must be before end_date ({self.end_date})"
            )

        logger.info(
            f"Initialized ingestion for {self.ticker} from {self.start_date.date()} to {self.end_date.date()}"
        )

    @staticmethod
    def _parse_date(date: str | datetime) -> datetime:
        """Parse date string or datetime object.

        Args:
            date: Date as string (YYYY-MM-DD) or datetime object.

        Returns:
            datetime object.

        Raises:
            ValueError: If date string format is invalid.
        """
        if isinstance(date, datetime):
            return date

        try:
            return datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {date}. Expected YYYY-MM-DD")

    def download(self, progress: bool = False) -> pd.DataFrame:
        """Download stock data from Yahoo Finance.

        Args:
            progress: If True, show download progress bar.

        Returns:
            DataFrame with OHLCV data (Open, High, Low, Close, Volume).

        Raises:
            ValueError: If download fails or returns empty data.

        Example:
            >>> df = ingestion.download()
            >>> print(df.columns)
            Index(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
        """
        logger.info(f"Downloading data for {self.ticker}...")

        try:
            df = yf.download(
                self.ticker,
                start=self.start_date.strftime("%Y-%m-%d"),
                end=self.end_date.strftime("%Y-%m-%d"),
                progress=progress,
            )

            if df.empty:
                raise ValueError(f"No data downloaded for {self.ticker}. Check ticker symbol.")

            # Simplify multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            logger.info(
                f"Downloaded {len(df)} records from {df.index[0].date()} to {df.index[-1].date()}"
            )
            logger.debug(f"Columns: {df.columns.tolist()}")

            return df

        except Exception as e:
            logger.error(f"Failed to download data for {self.ticker}: {e}")
            raise

    def download_and_validate(self, progress: bool = False) -> pd.DataFrame:
        """Download data and perform basic validation checks.

        Args:
            progress: If True, show download progress bar.

        Returns:
            Validated DataFrame.

        Raises:
            ValueError: If validation fails.
        """
        df = self.download(progress=progress)

        # Validate required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check for NaN values
        nan_counts = df.isnull().sum()
        if nan_counts.any():
            logger.warning(f"Found NaN values: {nan_counts[nan_counts > 0].to_dict()}")

        # Check for negative prices (data quality issue)
        price_cols = ["Open", "High", "Low", "Close"]
        for col in price_cols:
            if (df[col] < 0).any():
                logger.error(f"Found negative prices in column {col}")
                raise ValueError(f"Invalid data: negative prices in {col}")

        # Check for zero volume days
        zero_volume_days = (df["Volume"] == 0).sum()
        if zero_volume_days > 0:
            logger.warning(f"Found {zero_volume_days} days with zero volume")

        logger.info("Data validation passed ✓")
        return df

    def get_ticker_info(self) -> dict:
        """Get additional information about the ticker.

        Returns:
            Dictionary with ticker metadata (name, sector, market cap, etc.).

        Example:
            >>> info = ingestion.get_ticker_info()
            >>> print(info.get("longName"))
        """
        try:
            ticker_obj = yf.Ticker(self.ticker)
            info = ticker_obj.info
            logger.debug(f"Retrieved info for {self.ticker}")
            return info
        except Exception as e:
            logger.warning(f"Could not retrieve ticker info: {e}")
            return {}

    def save_to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """Save DataFrame to CSV file.

        Args:
            df: DataFrame to save.
            filepath: Path to save CSV file.
        """
        df.to_csv(filepath)
        logger.info(f"Data saved to {filepath}")


def download_multiple_tickers(
    tickers: list[str], start_date: str | datetime, end_date: Optional[str | datetime] = None
) -> dict[str, pd.DataFrame]:
    """Download data for multiple tickers.

    Args:
        tickers: List of ticker symbols.
        start_date: Start date for all tickers.
        end_date: End date for all tickers.

    Returns:
        Dictionary mapping ticker to DataFrame.

    Example:
        >>> data = download_multiple_tickers(["PETR4.SA", "VALE3.SA"], "2020-01-01")
        >>> print(data.keys())
    """
    results = {}

    for ticker in tickers:
        try:
            ingestion = StockDataIngestion(ticker, start_date, end_date)
            df = ingestion.download_and_validate()
            results[ticker] = df
        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")

    return results
