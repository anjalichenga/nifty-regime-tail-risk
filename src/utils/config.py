"""Configuration loader and schema validator."""
import hashlib
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    primary_ticker: str = "^NSEI"
    optional_tickers: list[str] = Field(default_factory=list)
    start_date: str = "2007-09-01"
    end_date: str | None = None
    auto_adjust: bool = False
    availability_lags: dict[str, int] = Field(default_factory=dict)
    min_volume_coverage: float = 0.90
    max_ffill_limit: int = 3

class FeaturesConfig(BaseModel):
    annualization_factor: float = 15.874507866387544
    rv_windows: list[int] = Field(default_factory=lambda: [5, 20, 60])
    parkinson_window: int = 20
    downside_window: int = 20
    drawdown_window: int = 252
    momentum_windows: list[int] = Field(default_factory=lambda: [20, 60])
    sma_short: int = 50
    sma_long: int = 200
    vol_z_window: int = 252
    volume_z_window: int = 60
    corr_window: int = 60
    vix_change_window: int = 20

class ArimaConfig(BaseModel):
    p_max: int = 3
    q_max: int = 3
    include_constant: bool = True
    criterion: str = "bic"

class GarchConfig(BaseModel):
    p: int = 1
    q: int = 1
    o: int = 0
    dist: str = "t"

class HmmConfig(BaseModel):
    k_list: list[int] = Field(default_factory=lambda: [2, 3, 4, 5])
    default_k: int = 3
    covariance_type: str = "full"
    n_init: int = 10
    min_covar: float = 0.001
    min_occupancy: float = 0.02
    features: list[str] = Field(default_factory=lambda: ["return", "ln_rv20", "drawdown"])

class IsolationForestConfig(BaseModel):
    n_estimators: int = 300
    contamination: str = "auto"
    percentile_threshold: float = 0.99
    features: list[str] = Field(default_factory=lambda: [
        "return", "ln_rv20", "vol_ratio_5_20", "drawdown_252", "momentum_20", "ln_parkinson"
    ])

class PeltConfig(BaseModel):
    model: str = "normal"
    min_size: int = 40
    penalty_multiplier: float = 3.0

class CusumConfig(BaseModel):
    k: float = 0.5
    h: float = 5.0

class ChangepointsConfig(BaseModel):
    pelt: PeltConfig = Field(default_factory=PeltConfig)
    cusum: CusumConfig = Field(default_factory=CusumConfig)

class EvtConfig(BaseModel):
    q_u: float = 0.90
    q_u_sensitivity: list[float] = Field(default_factory=lambda: [0.85, 0.95])
    min_exceedances: int = 50
    risk_quantiles: list[float] = Field(default_factory=lambda: [0.95, 0.99])

class ModelsConfig(BaseModel):
    seed: int = 42
    initial_train_days: int = 1260
    refit_cadence: int = 63
    arima_reselect_cadence: int = 252
    dev_end_date: str = "2014-12-31"
    test_start_date: str = "2015-01-01"
    arima: ArimaConfig = Field(default_factory=ArimaConfig)
    garch: GarchConfig = Field(default_factory=GarchConfig)
    hmm: HmmConfig = Field(default_factory=HmmConfig)
    isolation_forest: IsolationForestConfig = Field(default_factory=IsolationForestConfig)
    changepoints: ChangepointsConfig = Field(default_factory=ChangepointsConfig)
    evt: EvtConfig = Field(default_factory=EvtConfig)

class StressIndexConfig(BaseModel):
    headline_method: str = "W-EQ"
    methods: list[str] = Field(default_factory=lambda: ["W-EQ", "W-PCA", "W-LOGIT"])
    bands: dict[str, list[float]] = Field(default_factory=dict)
    alarm_threshold_primary: float = 50.0
    alarm_threshold_secondary: float = 75.0

class BacktestConfig(BaseModel):
    event_h: int = 10
    event_x: float = 0.05
    refractory_period: int = 20
    spell_max_gap: int = 5
    sensitivity_h: list[int] = Field(default_factory=lambda: [5, 10, 20])
    sensitivity_x: list[float] = Field(default_factory=lambda: [0.03, 0.05, 0.07])
    bootstrap_resamples: int = 1000
    bootstrap_block_size: int = 20

class AppConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    stress_index: StressIndexConfig = Field(default_factory=StressIndexConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

def load_config(config_path: Path | None = None) -> AppConfig:
    """Loads and validates configuration from YAML."""
    if config_path is None:
        from src.utils.paths import CONFIGS_DIR
        config_path = CONFIGS_DIR / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f)
    return AppConfig.model_validate(raw_dict)

def compute_config_hash(config: AppConfig) -> str:
    """Computes a SHA256 hex digest of the normalized config model."""
    serialized = config.model_dump_json(indent=None)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
