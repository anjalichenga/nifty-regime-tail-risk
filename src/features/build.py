"""Master feature dataset builder."""

import pandas as pd

from src.features.cross_asset import compute_cross_asset_features
from src.features.drawdown import compute_drawdowns
from src.features.returns import compute_returns
from src.features.technical import compute_technical_features
from src.features.volatility import compute_volatility_features
from src.utils.config import AppConfig
from src.utils.logging import get_logger
from src.utils.paths import DATA_PROCESSED_DIR

logger = get_logger(__name__)

def build_feature_dataset(
    primary_df: pd.DataFrame,
    secondary_dfs: dict[str, pd.DataFrame] | None = None,
    config: AppConfig | None = None,
    volume_enabled: bool = True,
) -> pd.DataFrame:
    """
    Constructs the full feature matrix from aligned price series.
    Guarantees causal rolling calculations and no-lookahead features.
    """
    if secondary_dfs is None:
        secondary_dfs = {}

    close = primary_df["Close"]
    high = primary_df["High"] if "High" in primary_df.columns else None
    low = primary_df["Low"] if "Low" in primary_df.columns else None
    volume = primary_df["Volume"] if "Volume" in primary_df.columns else None
    valid_ohlc = primary_df["valid_ohlc"] if "valid_ohlc" in primary_df.columns else None

    # 1. Returns and momentum
    ret_df = compute_returns(close)

    # 2. Drawdowns
    dd_df = compute_drawdowns(close, rolling_window=252)

    # 3. Volatilities
    vol_df = compute_volatility_features(
        returns=ret_df["return"],
        high=high,
        low=low,
        valid_ohlc=valid_ohlc,
        annualization_factor=15.874507866387544,
    )

    # 4. Technicals & volume
    tech_df = compute_technical_features(
        close=close,
        volume=volume,
        volume_enabled=volume_enabled,
        sma_short=50,
        sma_long=200,
        volume_window=60,
    )

    # 5. Cross-asset features
    cross_df = compute_cross_asset_features(
        primary_returns=ret_df["return"],
        secondary_dfs=secondary_dfs,
        corr_window=60,
        vix_change_window=20,
    )

    # Combine all
    base_cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "valid_ohlc"] if c in primary_df.columns]
    base_df = primary_df[base_cols]
    master_df = pd.concat([base_df, ret_df, dd_df, vol_df, tech_df, cross_df], axis=1)
    master_df = master_df.loc[:, ~master_df.columns.duplicated(keep="first")]

    # Save to data/processed/features.parquet
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED_DIR / "features.parquet"
    master_df.to_parquet(out_path, engine="pyarrow", index=True)
    logger.info("Successfully built and saved feature matrix with %d rows, %d columns to %s",
                len(master_df), master_df.shape[1], out_path.name)

    return master_df
