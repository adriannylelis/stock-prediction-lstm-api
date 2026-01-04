"""Feature engineering - Technical indicators for stock data.

This module calculates technical indicators commonly used in stock market analysis.
Indicators include: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, etc.
"""

import numpy as np
import pandas as pd
from loguru import logger


class TechnicalIndicators:
    """Calculator for technical indicators.

    This class provides methods to calculate various technical indicators
    that can be used as features for LSTM model training.

    All indicators are calculated using pandas and numpy for performance.

    Example:
        >>> ti = TechnicalIndicators(df)
        >>> df_enriched = ti.add_all_indicators()
        >>> print(df_enriched.columns)
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize with price data.

        Args:
            df: DataFrame with OHLCV columns (Open, High, Low, Close, Volume).

        Raises:
            ValueError: If required columns are missing.
        """
        required_cols = ["Close"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.df = df.copy()
        logger.debug(f"Initialized TechnicalIndicators with {len(df)} rows")

    def add_sma(self, windows: list[int] = [20, 50, 200]) -> pd.DataFrame:
        """Add Simple Moving Averages (SMA).

        SMA is the average price over a specified number of periods.

        Args:
            windows: List of window sizes (in days) for SMA calculation.

        Returns:
            DataFrame with added SMA columns.

        Example:
            >>> df = ti.add_sma([20, 50])  # SMA_20, SMA_50
        """
        for window in windows:
            col_name = f"SMA_{window}"
            self.df[col_name] = self.df["Close"].rolling(window=window).mean()
            logger.debug(f"Added {col_name}")

        return self.df

    def add_ema(self, windows: list[int] = [12, 26]) -> pd.DataFrame:
        """Add Exponential Moving Averages (EMA).

        EMA gives more weight to recent prices compared to SMA.

        Args:
            windows: List of window sizes for EMA calculation.

        Returns:
            DataFrame with added EMA columns.

        Example:
            >>> df = ti.add_ema([12, 26])  # EMA_12, EMA_26
        """
        for window in windows:
            col_name = f"EMA_{window}"
            self.df[col_name] = self.df["Close"].ewm(span=window, adjust=False).mean()
            logger.debug(f"Added {col_name}")

        return self.df

    def add_rsi(self, period: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index (RSI).

        RSI measures the magnitude of recent price changes to evaluate
        overbought or oversold conditions. Range: 0-100.

        Args:
            period: Number of periods for RSI calculation (typically 14).

        Returns:
            DataFrame with added RSI column.

        Example:
            >>> df = ti.add_rsi(14)  # RSI_14
        """
        # Calculate price changes
        delta = self.df["Close"].diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        col_name = f"RSI_{period}"
        self.df[col_name] = rsi
        logger.debug(f"Added {col_name}")

        return self.df

    def add_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Add MACD (Moving Average Convergence Divergence).

        MACD shows the relationship between two EMAs and is used to identify
        trend changes.

        Args:
            fast: Fast EMA period (typically 12).
            slow: Slow EMA period (typically 26).
            signal: Signal line period (typically 9).

        Returns:
            DataFrame with added MACD, MACD_signal, and MACD_hist columns.

        Example:
            >>> df = ti.add_macd()  # MACD, MACD_signal, MACD_hist
        """
        # Calculate EMAs
        ema_fast = self.df["Close"].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df["Close"].ewm(span=slow, adjust=False).mean()

        # MACD line
        macd = ema_fast - ema_slow
        self.df["MACD"] = macd

        # Signal line
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        self.df["MACD_signal"] = macd_signal

        # MACD histogram
        self.df["MACD_hist"] = macd - macd_signal

        logger.debug(f"Added MACD ({fast}, {slow}, {signal})")

        return self.df

    def add_bollinger_bands(self, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """Add Bollinger Bands.

        Bollinger Bands consist of a middle band (SMA) and upper/lower bands
        that are standard deviations away from the middle band.

        Args:
            window: Window size for SMA calculation.
            num_std: Number of standard deviations for bands.

        Returns:
            DataFrame with BB_upper, BB_middle, BB_lower columns.

        Example:
            >>> df = ti.add_bollinger_bands(20, 2)
        """
        # Middle band (SMA)
        sma = self.df["Close"].rolling(window=window).mean()
        std = self.df["Close"].rolling(window=window).std()

        self.df["BB_middle"] = sma
        self.df["BB_upper"] = sma + (num_std * std)
        self.df["BB_lower"] = sma - (num_std * std)

        logger.debug(f"Added Bollinger Bands ({window}, {num_std}σ)")

        return self.df

    def add_atr(self, period: int = 14) -> pd.DataFrame:
        """Add Average True Range (ATR).

        ATR measures market volatility. Higher ATR indicates higher volatility.

        Args:
            period: Number of periods for ATR calculation.

        Returns:
            DataFrame with ATR column.

        Raises:
            ValueError: If High, Low, Close columns are missing.

        Example:
            >>> df = ti.add_atr(14)  # ATR_14
        """
        required = ["High", "Low", "Close"]
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"ATR requires columns: {missing}")

        # Calculate True Range
        high_low = self.df["High"] - self.df["Low"]
        high_close = np.abs(self.df["High"] - self.df["Close"].shift())
        low_close = np.abs(self.df["Low"] - self.df["Close"].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # ATR is the moving average of True Range
        col_name = f"ATR_{period}"
        self.df[col_name] = true_range.rolling(window=period).mean()
        logger.debug(f"Added {col_name}")

        return self.df

    def add_returns(self) -> pd.DataFrame:
        """Add price returns (percentage change).

        Returns:
            DataFrame with Returns column.
        """
        self.df["Returns"] = self.df["Close"].pct_change()
        logger.debug("Added Returns")

        return self.df

    def add_all_indicators(
        self,
        sma_windows: list[int] = [20, 50, 200],
        ema_windows: list[int] = [12, 26],
        rsi_period: int = 14,
        macd_params: tuple[int, int, int] = (12, 26, 9),
        bb_params: tuple[int, float] = (20, 2.0),
        atr_period: int = 14,
    ) -> pd.DataFrame:
        """Add all technical indicators.

        Args:
            sma_windows: Windows for SMA.
            ema_windows: Windows for EMA.
            rsi_period: Period for RSI.
            macd_params: (fast, slow, signal) for MACD.
            bb_params: (window, num_std) for Bollinger Bands.
            atr_period: Period for ATR.

        Returns:
            DataFrame with all technical indicators.

        Example:
            >>> df_enriched = ti.add_all_indicators()
        """
        logger.info("Adding all technical indicators...")

        self.add_sma(sma_windows)
        self.add_ema(ema_windows)
        self.add_rsi(rsi_period)
        self.add_macd(*macd_params)
        self.add_bollinger_bands(*bb_params)

        # ATR requires OHLC data
        if all(col in self.df.columns for col in ["High", "Low", "Close"]):
            self.add_atr(atr_period)
        else:
            logger.warning("Skipping ATR: missing OHLC columns")

        self.add_returns()

        logger.info(f"Added technical indicators. New shape: {self.df.shape}")

        return self.df

    def fill_missing_values(self, method: str = "ffill") -> pd.DataFrame:
        """Fill missing values created by indicator calculations.

        Args:
            method: Filling method ('ffill', 'bfill', 'mean', 'zero').

        Returns:
            DataFrame with filled missing values.
        """
        nan_before = self.df.isnull().sum().sum()

        if method == "ffill":
            self.df = self.df.ffill().bfill()
        elif method == "bfill":
            self.df = self.df.bfill().ffill()
        elif method == "mean":
            self.df = self.df.fillna(self.df.mean())
        elif method == "zero":
            self.df = self.df.fillna(0)
        else:
            raise ValueError(f"Invalid method: {method}")

        nan_after = self.df.isnull().sum().sum()
        logger.info(f"Filled {nan_before - nan_after} missing values using {method}")

        return self.df
