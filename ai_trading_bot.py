#!/usr/bin/env python3
"""
walk_ai_bot_v2_patched_fixed.py - Fixed full bot (complete file)

Fixes applied:
 - Proper ATR retrieval for SL/TP instead of synthetic zero-ATR.
 - Enforce broker symbol constraints: trade_stops_level, digits, point, volume_min/step/volume_max.
 - Robust volume calculation using symbol contract size & point.
 - Fixed backtester probability filtering logic.
 - Added device logging and a few defensive checks.
 - Improved logging around order requests and results.
 - Kept original architecture, models, and training flow intact.
"""
from __future__ import annotations
import os
import sys
import time
import math
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd

# Torch for LSTM
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
import joblib

# optional xgboost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    xgb = None
    XGB_AVAILABLE = False

# MT5
try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

# ---------- Logging ----------
LOGFILE = os.environ.get('WALK_AI_LOG', 'logs/walk_ai_bot_v2_patched_fixed.log')
os.makedirs(os.path.dirname(LOGFILE), exist_ok=True)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(LOGFILE, encoding='utf-8')])
logger = logging.getLogger('walk_ai_bot_v2_patched_fixed')

# ---------- Config ----------
@dataclass
class Config:
    SYMBOL: str = "XAUUSDm"
    TIMEFRAME: str = 'M5'
    M5_PERIOD_MINUTES: int = 5
    HISTORY_BARS: int = 600000
    LOOKBACK_BARS: int = 24
    PRED_HORIZON: int = 3
    ATR_PERIOD: int = 14
    LSTM_HIDDEN: int = 64
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_DIR: str = 'models_v2'
    MIN_TRADE_PROB: float = 0.55  # aggressiveness for scalping
    PROB_ONLY_THRESHOLD: float = 0.85  # very strong prob-only trigger (used in score gating)
    MAX_POSITIONS: int = 1
    RISK_PER_TRADE: float = 0.01
    MAX_VOLUME: float = 0.01
    SPREAD_PIPS_WARNING: float = 0.20
    TRADE_TIMEOUT_BARS: int = 6
    SIM_SPREAD: float = 0.17
    SIM_SLIPPAGE_PIPS: float = 0.02
    USE_XGB: bool = True
    PROBABILITY_WEIGHT_TREE: float = 0.65
    PROBABILITY_WEIGHT_LSTM: float = 0.35
    PROB_SHARPEN: float = 1.15
    TRAIN_TEST_SPLIT_RATIO: float = 0.8
    SEED: int = 42
    STRATEGIES: List[str] = field(default_factory=lambda: ['momentum', 'reversion', 'squeeze'])
    LABEL_THR_FACTORS: Dict[str, float] = field(default_factory=lambda: {'momentum': 0.06, 'reversion': 0.12, 'squeeze': 0.06})
    STRATEGY_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {'momentum': 1.0, 'reversion': 1.0, 'squeeze': 1.0})
    SCALER_REFRESH_LOOPS: int = 300
    LIVE_LOOP_DELAY: float = 1.0

np.random.seed(Config().SEED)

# ---------- Numerically-stable helpers ----------
def _safe_clip(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Coerce to float, replace non-finite (nan, inf) with neutral 0.5,
    and clip into (eps, 1-eps).
    """
    p = np.asarray(p, dtype=float)             # ensure numeric array
    # replace non-finite entries (nan/inf) with neutral probability 0.5
    p = np.where(np.isfinite(p), p, 0.5)
    return np.clip(p, eps, 1.0 - eps)


def logit_np(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Numerically-stable logit:
      - safe clip probabilities away from 0/1
      - compute logit under an errstate to avoid noisy runtime warnings
      - map any remaining non-finite logits (if any) to large finite numbers,
        preserving sign (positive for p>0.5, negative for p<0.5), or 0 if neutral.
    """
    p = _safe_clip(p, eps)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.log(p / (1.0 - p))
    # replace non-finite outputs (should be rare) with large finite numbers
    sign = np.sign(p - 0.5)         # >0 => positive logit, <0 => negative, 0 => neutral
    out = np.where(np.isfinite(out), out, sign * 1e6)
    return out

def expit_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return 1.0 / (1.0 + np.exp(-x))

# ---------- Utilities ----------
def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()

def compute_rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def compute_bbw(close: pd.Series, window: int) -> pd.Series:
    ma = close.rolling(window=window, min_periods=1).mean()
    std = close.rolling(window=window, min_periods=1).std().fillna(0.0)
    bbw = (2 * std) / ma.replace(0, np.nan)
    return bbw.fillna(0.0)

def hurst_exponent(ts: np.ndarray) -> float:
    N = len(ts)
    if N < 20:
        return 0.5
    lags = np.floor(np.logspace(1, np.log10(N / 2), num=20)).astype(int)
    tau = []
    for lag in lags:
        pp = np.subtract(ts[lag:], ts[:-lag])
        tau.append(np.sqrt(np.std(pp)))
    poly = np.polyfit(np.log(lags + 1e-8), np.log(np.array(tau) + 1e-8), 1)
    return float(poly[0])

def kalman_smooth(series: pd.Series) -> pd.Series:
    n = len(series)
    xhat = np.zeros(n)
    P = np.zeros(n)
    Q = 1e-5
    R = np.maximum(np.var(series.dropna()) * 0.01, 1e-8)
    xhat[0] = series.iloc[0]
    P[0] = 1.0
    for k in range(1, n):
        xhatminus = xhat[k - 1]
        Pminus = P[k - 1] + Q
        K = Pminus / (Pminus + R)
        xhat[k] = xhatminus + K * (series.iloc[k] - xhatminus)
        P[k] = (1 - K) * Pminus
    return pd.Series(xhat, index=series.index)

# ---------- Data / MT5 connector ----------
class MT5Connector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.connected = False

    def connect(self) -> bool:
        if mt5 is None:
            logger.warning('MetaTrader5 package not available. Live mode disabled.')
            return False
        try:
            ok = mt5.initialize()
            if not ok:
                logger.warning('MT5 initialize returned False')
                return False
            self.connected = True
            logger.info('MT5 initialized')
            return True
        except Exception as e:
            logger.exception('Exception initializing MT5: %s', e)
            self.connected = False
            return False

    def disconnect(self):
        if mt5 is not None and self.connected:
            try:
                mt5.shutdown()
                self.connected = False
            except Exception:
                pass

    def fetch_history_m5(self, bars: int) -> pd.DataFrame:
        if mt5 is None:
            raise RuntimeError('MT5 not available')
        TF = getattr(mt5, 'TIMEFRAME_M5', 5)
        rates = mt5.copy_rates_from_pos(self.cfg.SYMBOL, TF, 0, bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f'Failed to fetch rates for {self.cfg.SYMBOL}')
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def fetch_recent_ticks(self, seconds_back: int = 60) -> pd.DataFrame:
        return self.fetch_history_m5(200)

# ---------- Feature engineering ----------
class FeatureEngineer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scaler = StandardScaler()

    def feature_columns(self) -> List[str]:
        return ['close', 'close_k', 'ATR', 'RSI', 'EMA5', 'EMA20', 'BBW', 'hour_sin', 'hour_cos',
                'zret', 'vol_norm', 'hurst', 'rsi_delta', 'atr_log',
                'spread_proxy', 'spread_rel'] + [f'lag{l}' for l in range(1, 6)]

    def build_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
        df = df.copy().reset_index(drop=True)
        df['time'] = pd.to_datetime(df['time'])
        df['log_ret'] = np.log(df['close']).diff().fillna(0.0)
        df['ATR'] = compute_atr(df['high'], df['low'], df['close'], self.cfg.ATR_PERIOD)
        df['RSI'] = compute_rsi(df['close'], 14)
        df['EMA5'] = compute_ema(df['close'], 5)
        df['EMA20'] = compute_ema(df['close'], 20)
        df['BBW'] = compute_bbw(df['close'], 20)

        df['spread_proxy'] = (df['high'] - df['low']).clip(lower=1e-6)
        df['spread_rel'] = df['spread_proxy'] / df['ATR'].replace(0, np.nan)

        df['hour'] = df['time'].dt.hour + df['time'].dt.minute / 60.0
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

        for l in range(1, 6):
            df[f'lag{l}'] = df['log_ret'].shift(l).fillna(0.0)

        df['zret'] = (df['log_ret'] - df['log_ret'].rolling(200).mean()).fillna(0.0)
        df['vol_norm'] = df['ATR'] / df['ATR'].rolling(200).mean().replace(0, np.nan).fillna(df['ATR'].mean())

        df['hurst'] = 0.5
        try:
            hurst_val = hurst_exponent(df['close'].values[-500:])
            df['hurst'] = hurst_val
        except Exception:
            pass

        try:
            df['close_k'] = kalman_smooth(df['close'])
        except Exception:
            df['close_k'] = df['close']

        df['rsi_delta'] = df['RSI'].diff().fillna(0.0)
        df['atr_log'] = np.log(df['ATR'].replace(0, 1e-6))

        df = df.dropna().reset_index(drop=True)
        feature_cols = self.feature_columns()
        X = df[feature_cols].values.astype(np.float32)

        if fit_scaler:
            self.scaler = StandardScaler()
            Xs = self.scaler.fit_transform(X)
        else:
            if not hasattr(self.scaler, 'scale_'):
                raise ValueError("FeatureEngineer.scaler is not fitted. Save/load fe_scaler.pkl or call build_features(..., fit_scaler=True).")
            Xs = self.scaler.transform(X)

        return df, Xs

# ---------- Multi-strategy label construction ----------
def construct_labels_and_rewards(df: pd.DataFrame, cfg: Config):
    n = len(df)
    atr = df['ATR'].values
    closes = df['close'].values
    horizon = cfg.PRED_HORIZON
    labels = {s: np.full(n, -1, dtype=np.int8) for s in cfg.STRATEGIES}
    rewards = {s: np.full(n, np.nan, dtype=float) for s in cfg.STRATEGIES}

    ema20 = compute_ema(df['close'], 20)
    bbw = df['BBW'].values
    bbw_thresh = np.percentile(bbw, 30) if len(bbw) > 10 else np.nan

    spread_cost_global = cfg.SIM_SPREAD
    slippage = cfg.SIM_SLIPPAGE_PIPS

    for i in range(n - horizon):
        future_slice = closes[i + 1: i + horizon + 1]
        future_max = np.max(future_slice)
        future_min = np.min(future_slice)

        if 'spread_proxy' in df.columns:
            effective_spread = float(df['spread_proxy'].iloc[i])
        else:
            effective_spread = float(spread_cost_global)

        realized_long = (future_max - closes[i]) - effective_spread - slippage
        realized_short = (closes[i] - future_min) - effective_spread - slippage

        # momentum
        thr_m = atr[i] * cfg.LABEL_THR_FACTORS.get('momentum', 0.06)
        if realized_long >= thr_m and realized_long > realized_short:
            labels['momentum'][i] = 1
            rewards['momentum'][i] = realized_long / max(1e-6, atr[i])
        elif realized_short >= thr_m and realized_short > realized_long:
            labels['momentum'][i] = 0
            rewards['momentum'][i] = realized_short / max(1e-6, atr[i])
        else:
            labels['momentum'][i] = -1
            rewards['momentum'][i] = np.nan

        # reversion
        thr_r = atr[i] * cfg.LABEL_THR_FACTORS.get('reversion', 0.12)
        dev = closes[i] - ema20.iloc[i]
        if abs(dev) >= thr_r:
            if dev > 0 and (closes[i] - future_min) >= thr_r * 0.5:
                labels['reversion'][i] = 0
                pnl = (closes[i] - future_min) - effective_spread - slippage
                rewards['reversion'][i] = pnl / max(1e-6, atr[i])
            elif dev < 0 and (future_max - closes[i]) >= thr_r * 0.5:
                labels['reversion'][i] = 1
                pnl = (future_max - closes[i]) - effective_spread - slippage
                rewards['reversion'][i] = pnl / max(1e-6, atr[i])
            else:
                labels['reversion'][i] = -1
                rewards['reversion'][i] = np.nan
        else:
            labels['reversion'][i] = -1
            rewards['reversion'][i] = np.nan

        # squeeze
        thr_s = atr[i] * cfg.LABEL_THR_FACTORS.get('squeeze', 0.06)
        if not np.isnan(bbw_thresh) and bbw[i] < bbw_thresh:
            if realized_long >= thr_s and realized_long > realized_short:
                labels['squeeze'][i] = 1
                rewards['squeeze'][i] = realized_long / max(1e-6, atr[i])
            elif realized_short >= thr_s and realized_short > realized_long:
                labels['squeeze'][i] = 0
                rewards['squeeze'][i] = realized_short / max(1e-6, atr[i])
            else:
                labels['squeeze'][i] = -1
                rewards['squeeze'][i] = np.nan
        else:
            labels['squeeze'][i] = -1
            rewards['squeeze'][i] = np.nan

    for s in cfg.STRATEGIES:
        labels[s][-horizon:] = -1
        rewards[s][-horizon:] = np.nan

    return labels, rewards

def composite_label_from_multi(labels_multi: Dict[str, np.ndarray], cfg: Config) -> np.ndarray:
    n = len(next(iter(labels_multi.values())))
    comp = np.full(n, -1, dtype=np.int8)
    for i in range(n):
        votes = [labels_multi[s][i] for s in cfg.STRATEGIES]
        if any(v == 1 for v in votes):
            comp[i] = 1
        elif any(v == 0 for v in votes):
            comp[i] = 0
        else:
            comp[i] = -1
    return comp

# ---------- PyTorch LSTM ----------
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, idx):
        x = self.X[idx: idx + self.seq_len]
        y = self.y[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        logits = self.fc(out).squeeze(-1)
        return logits

# ---------- Model manager ----------
class ModelManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tree_models: Dict[str, Any] = {}
        self.lstm_models: Dict[str, Any] = {}
        self.lstm_scalers: Dict[str, Any] = {}
        os.makedirs(self.cfg.MODEL_DIR, exist_ok=True)

    def train_tree(self, X: np.ndarray, y: np.ndarray, strat: str) -> None:
        mask = y != -1
        Xf = X[mask]
        yf = y[mask]
        if len(yf) < 50:
            logger.warning('Not enough samples for tree training for strat=%s (n=%d)', strat, len(yf))
            return
        if XGB_AVAILABLE and self.cfg.USE_XGB:
            logger.info('Training XGBoost classifier for strat=%s', strat)
            dmat = xgb.DMatrix(Xf, label=yf)
            params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'seed': self.cfg.SEED}
            bst = xgb.train(params, dmat, num_boost_round=200, verbose_eval=False)
            self.tree_models[strat] = bst
            joblib.dump(bst, os.path.join(self.cfg.MODEL_DIR, f'tree_{strat}.pkl'))
        else:
            logger.info('Training RandomForest classifier for strat=%s', strat)
            rf = RandomForestClassifier(n_estimators=200, random_state=self.cfg.SEED, n_jobs=-1)
            rf.fit(Xf, yf)
            calibrated = CalibratedClassifierCV(rf, method='sigmoid', cv=3)
            calibrated.fit(Xf, yf)
            self.tree_models[strat] = calibrated
            joblib.dump(calibrated, os.path.join(self.cfg.MODEL_DIR, f'tree_{strat}.pkl'))

    def predict_tree_proba(self, X: np.ndarray, strat: str) -> np.ndarray:
        if strat not in self.tree_models:
            raise RuntimeError(f'Tree model for {strat} not trained')
        model = self.tree_models[strat]
        if XGB_AVAILABLE and isinstance(model, xgb.Booster):
            dm = xgb.DMatrix(X)
            p = model.predict(dm)
            return np.vstack([1 - p, p]).T[:, 1]
        else:
            return model.predict_proba(X)[:, 1]

    def train_lstm(self, X: np.ndarray, y: np.ndarray, seq_len: int, strat: str):
        mask = y != -1
        Xf = X[mask]
        yf = y[mask]
        if len(yf) < seq_len + 50:
            logger.warning('Not enough samples for LSTM training strat=%s', strat)
            return
        if len(yf) < 200:
            epochs = 40
            batch_size = 16
        else:
            epochs = 15
            batch_size = 64
        scaler = MinMaxScaler()
        Xs = scaler.fit_transform(Xf)
        ds = SeqDataset(Xs, yf, seq_len)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        model = LSTMClassifier(input_size=X.shape[1], hidden_size=self.cfg.LSTM_HIDDEN).to(self.cfg.DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.BCEWithLogitsLoss()
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for bx, by in loader:
                bx, by = bx.to(self.cfg.DEVICE), by.to(self.cfg.DEVICE)
                opt.zero_grad()
                logits = model(bx)
                loss = crit(logits, by)
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(bx)
            logger.info('LSTM strat=%s epoch %d loss %.6f', strat, epoch, total_loss / max(1, len(ds)))
        self.lstm_models[strat] = model
        self.lstm_scalers[strat] = scaler
        torch.save(model.state_dict(), os.path.join(self.cfg.MODEL_DIR, f'lstm_{strat}.pt'))
        joblib.dump(scaler, os.path.join(self.cfg.MODEL_DIR, f'lstm_scaler_{strat}.pkl'))
        logger.info('Saved LSTM and scaler for strat=%s', strat)

    def train_reward_regressor(self, X: np.ndarray, r: np.ndarray, strat: str):
        mask = ~np.isnan(r)
        Xf = X[mask]
        rf = r[mask]
        if len(rf) < 50:
            logger.warning('Not enough reward samples for strat=%s (n=%d)', strat, len(rf))
            return
        if XGB_AVAILABLE and self.cfg.USE_XGB:
            logger.info('Training XGBoost regressor for reward strat=%s', strat)
            dtr = xgb.DMatrix(Xf, label=rf)
            params = {'objective': 'reg:squarederror', 'seed': self.cfg.SEED}
            bst = xgb.train(params, dtr, num_boost_round=200, verbose_eval=False)
            self.tree_models[f'reward_{strat}'] = bst
            joblib.dump(bst, os.path.join(self.cfg.MODEL_DIR, f'reward_{strat}.pkl'))
        else:
            logger.info('Training RandomForest regressor for reward strat=%s', strat)
            rfr = RandomForestRegressor(n_estimators=200, random_state=self.cfg.SEED, n_jobs=-1)
            rfr.fit(Xf, rf)
            self.tree_models[f'reward_{strat}'] = rfr
            joblib.dump(rfr, os.path.join(self.cfg.MODEL_DIR, f'reward_{strat}.pkl'))
        logger.info('Saved reward regressor for strat=%s', strat)

    def predict_reward(self, X: np.ndarray, strat: str) -> np.ndarray:
        key = f'reward_{strat}'
        if key not in self.tree_models:
            return np.zeros(X.shape[0], dtype=float)
        model = self.tree_models[key]
        if XGB_AVAILABLE and isinstance(model, xgb.Booster):
            dm = xgb.DMatrix(X)
            return model.predict(dm)
        else:
            return model.predict(X)

    def predict_lstm_proba(self, seq_tail: np.ndarray, strat: str) -> np.ndarray:
        if strat not in self.lstm_models or strat not in self.lstm_scalers:
            raise RuntimeError(f'LSTM or scaler for strat={strat} not available')
        scaler = self.lstm_scalers[strat]
        model = self.lstm_models[strat]
        st = scaler.transform(seq_tail)
        xb = torch.tensor(st.reshape(1, st.shape[0], st.shape[1]), dtype=torch.float32).to(self.cfg.DEVICE)
        model.eval()
        with torch.no_grad():
            logits = model(xb)
            prob = float(torch.sigmoid(logits).cpu().numpy().ravel()[0])
        return prob

    def predict_proba_multi(self, X: np.ndarray, seq_tail: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        res = {}
        for strat in self.cfg.STRATEGIES:
            tree_p = None
            lstm_p = None
            if strat in self.tree_models:
                try:
                    tree_p = self.predict_tree_proba(X, strat)
                except Exception as e:
                    logger.exception('Tree predict failed for strat=%s: %s', strat, e)
                    tree_p = None
            if strat in self.lstm_models and seq_tail is not None:
                try:
                    p = self.predict_lstm_proba(seq_tail, strat)
                    lstm_p = np.full(X.shape[0], p, dtype=float)
                except Exception as e:
                    logger.exception('LSTM predict failed for strat=%s: %s', strat, e)
                    lstm_p = None
            if tree_p is None and lstm_p is None:
                res[strat] = np.full(X.shape[0], 0.5, dtype=float)
                continue
            if tree_p is None:
                combined = lstm_p
            elif lstm_p is None:
                combined = tree_p
            else:
                lw = self.cfg.PROBABILITY_WEIGHT_TREE
                lw2 = self.cfg.PROBABILITY_WEIGHT_LSTM
                lt = logit_np(tree_p)
                ll = logit_np(lstm_p)
                comb_logit = (lw * lt + lw2 * ll) / (lw + lw2)
                combined = expit_np(comb_logit)
            try:
                comb_logit2 = logit_np(combined) * self.cfg.PROB_SHARPEN
                combined = expit_np(comb_logit2)
            except Exception:
                pass
            res[strat] = combined
        return res

    def predict_meta_proba(self, X: np.ndarray, seq_tail: Optional[np.ndarray] = None) -> np.ndarray:
        multi = self.predict_proba_multi(X, seq_tail=seq_tail)
        weights = np.array([self.cfg.STRATEGY_WEIGHTS.get(s, 1.0) for s in self.cfg.STRATEGIES], dtype=float)
        probs_stack = np.vstack([multi[s] for s in self.cfg.STRATEGIES])
        meta = np.average(probs_stack, axis=0, weights=weights)
        meta = expit_np(logit_np(meta) * self.cfg.PROB_SHARPEN)
        return meta

    def save(self):
        os.makedirs(self.cfg.MODEL_DIR, exist_ok=True)
        for strat, m in self.tree_models.items():
            joblib.dump(m, os.path.join(self.cfg.MODEL_DIR, f'tree_{strat}.pkl'))
        for strat, m in self.lstm_models.items():
            torch.save(m.state_dict(), os.path.join(self.cfg.MODEL_DIR, f'lstm_{strat}.pt'))
        for strat, s in self.lstm_scalers.items():
            joblib.dump(s, os.path.join(self.cfg.MODEL_DIR, f'lstm_scaler_{strat}.pkl'))
        try:
            joblib.dump(self.cfg, os.path.join(self.cfg.MODEL_DIR, 'config.pkl'))
        except Exception:
            pass
        logger.info('✅ ModelManager saved successfully.')

    @staticmethod
    def load(cfg: Config) -> 'ModelManager':
        mm = ModelManager(cfg)
        try:
            for strat in cfg.STRATEGIES:
                tree_path = os.path.join(cfg.MODEL_DIR, f'tree_{strat}.pkl')
                if os.path.exists(tree_path):
                    try:
                        mm.tree_models[strat] = joblib.load(tree_path)
                        logger.info('Loaded tree model for strat=%s', strat)
                    except Exception as e:
                        logger.warning('Failed loading tree model for %s: %s', strat, e)
                lstm_scaler_path = os.path.join(cfg.MODEL_DIR, f'lstm_scaler_{strat}.pkl')
                lstm_model_path = os.path.join(cfg.MODEL_DIR, f'lstm_{strat}.pt')
                if os.path.exists(lstm_scaler_path) and os.path.exists(lstm_model_path):
                    try:
                        scaler = joblib.load(lstm_scaler_path)
                        mm.lstm_scalers[strat] = scaler
                        n_features = getattr(scaler, 'n_features_in_', None)
                    except Exception as e:
                        logger.warning('Failed to load lstm scaler for %s: %s', strat, e)
                        n_features = None
                    try:
                        if n_features is None:
                            n_features = len(FeatureEngineer(cfg).feature_columns())
                        model = LSTMClassifier(input_size=int(n_features), hidden_size=cfg.LSTM_HIDDEN)
                        state_dict = torch.load(lstm_model_path, map_location=cfg.DEVICE)
                        try:
                            model.load_state_dict(state_dict)
                        except RuntimeError:
                            model.load_state_dict(state_dict, strict=False)
                        model.to(cfg.DEVICE)
                        mm.lstm_models[strat] = model
                        logger.info('Loaded LSTM for strat=%s', strat)
                    except Exception as e:
                        logger.warning('Could not load LSTM for %s: %s', strat, e)
            logger.info('✅ ModelManager loaded successfully.')
        except Exception as e:
            logger.exception('⚠️ Failed to load components: %s', e)
        return mm

# ---------- Backtesting engine ----------
class Backtester:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def simulate(self, df: pd.DataFrame, probs: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        results = []
        n = len(df)
        spread = self.cfg.SIM_SPREAD
        slippage = self.cfg.SIM_SLIPPAGE_PIPS
        for i in range(n - self.cfg.PRED_HORIZON):
            if labels[i] == -1:
                continue
            p = probs[i]
            # require a clear probability: either high long or high short
            if not (p >= self.cfg.MIN_TRADE_PROB or p <= (1.0 - self.cfg.MIN_TRADE_PROB)):
                continue
            # determine side using label confirmation and probability direction
            side = None
            if p >= 0.5 and labels[i] == 1:
                side = 1
            elif p < 0.5 and labels[i] == 0:
                side = -1
            else:
                # if label doesn't match but prob is extreme, allow prob-only direction
                if p >= self.cfg.PROB_ONLY_THRESHOLD:
                    side = 1
                elif p <= (1.0 - self.cfg.PROB_ONLY_THRESHOLD):
                    side = -1
                else:
                    side = None
            if side is None:
                continue
            entry = df['close'].iloc[i] + side * spread
            atr = df['ATR'].iloc[i]
            sl = entry - side * atr * 1.0
            tp = entry + side * atr * 0.6
            horizon_prices = df['close'].iloc[i + 1: i + self.cfg.PRED_HORIZON + 1].values
            exit_price = horizon_prices[-1] if len(horizon_prices) > 0 else entry
            for px in horizon_prices:
                if side == 1 and px >= tp:
                    exit_price = tp - slippage
                    break
                if side == -1 and px <= tp:
                    exit_price = tp + slippage
                    break
                if side == 1 and px <= sl:
                    exit_price = sl + slippage
                    break
                if side == -1 and px >= sl:
                    exit_price = sl - slippage
                    break
            pnl = (exit_price - entry) * side
            results.append({'index': i, 'prob': p, 'side': side, 'entry': entry, 'exit': exit_price, 'pnl': pnl})
        if not results:
            return {'trades': 0, 'win_rate': 0.0, 'net': 0.0, 'pnl_series': []}
        rdf = pd.DataFrame(results)
        wins = (rdf['pnl'] > 0).sum()
        net = rdf['pnl'].sum()
        return {'trades': len(rdf), 'win_rate': wins / len(rdf), 'net': net, 'pnl_series': rdf['pnl'].values}

# ---------- Execution / Live trading ----------
class RiskManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def compute_volume(self, account_balance: float, price: float, sl_price: float, symbol: Optional[str] = None) -> float:
        risk_amount = account_balance * self.cfg.RISK_PER_TRADE
        sl_distance = abs(price - sl_price)
        if sl_distance <= 0:
            return 0.01
        vol = 0.01
        try:
            if mt5 is not None and symbol is not None:
                sym = mt5.symbol_info(symbol)
            else:
                sym = None
            contract_size = getattr(sym, 'trade_contract_size', None)
            point = getattr(sym, 'point', None)
            if contract_size is not None and point is not None and point > 0:
                pip_value_per_lot = contract_size * point
                vol = risk_amount / max(1e-12, sl_distance * pip_value_per_lot)
            else:
                vol = risk_amount / max(1e-12, sl_distance * 1000.0)
        except Exception as e:
            logger.warning('Volume compute fallback due to: %s', e)
            vol = risk_amount / max(1e-12, sl_distance * 1000.0)
        vol = max(0.01, min(vol, self.cfg.MAX_VOLUME))
        return float(vol)

# Replace the existing ExecutorLive class with this one.

class ExecutorLive:
    """
    ExecutorLive (upgraded):
      - one aggregated position per direction
      - open larger lot but single position
      - partial-close system (configurable)
      - monitor open position, manage partial closes
      - regime-aware cooldown; ML can re-arm only after close + cooldown
    """
    def __init__(self, cfg: Config, models: ModelManager, fe: FeatureEngineer):
        self.cfg = cfg
        self.models = models
        self.fe = fe
        self.mt5 = MT5Connector(cfg)
        self.risk = RiskManager(cfg)

        # Execution state
        self.last_trade_time = 0.0
        self.last_side = None          # 'BUY' or 'SELL' or None
        self.in_cooldown_until = 0.0
        self.open_position_ticket = None
        self.open_position_type = None
        self.open_position_volume = 0.0
        self.partial_close_state = []  # list of dicts tracking executed partial closes

        # Configurable execution params
        self.MIN_SIGNAL_INTERVAL = 10.0        # minimal seconds between same-direction open attempts
        self.COOLDOWN_BASE_SEC = 300.0          # base cooldown seconds after full close
        self.REGIME_MULTIPLIER = {'trend': 0.5, 'meanrev': 1.5, 'chop': 2.0}
        # Partial close scheme: list of (fraction_to_close, tp_multiplier_of_atr)
        # Fractions apply to the volume remaining at the time of the partial close decision.
        # Example: first close 50% at tp=1.0*ATR, then close remaining 100% at tp=1.8*ATR
        self.PARTIAL_CLOSES = [
            {'frac': 0.5, 'tp_atr': 1.0},   # close 50% at +1.0 ATR
            {'frac': 1.0, 'tp_atr': 1.8}    # close remaining 100% at +1.8 ATR
        ]
        self.MIN_VOLUME_FOR_TRADE = 0.01

        self._stop = False

    def start(self, dry_run: bool = True):
        logger.info('ExecutorLive.start dry_run=%s device=%s', dry_run, self.cfg.DEVICE)
        if not dry_run and not self.mt5.connect():
            logger.error('MT5 connect failed; abort')
            return

        # Warm start: ensure scaler loaded or fitted
        try:
            hist = self.mt5.fetch_history_m5(self.cfg.HISTORY_BARS) if self.mt5.connected else None
            if hist is not None:
                self.fe.build_features(hist, fit_scaler=False)
        except Exception:
            pass

        loop_delay = max(0.5, getattr(self.cfg, 'LIVE_LOOP_DELAY', 1.0))

        while not self._stop:
            loop_start = time.time()
            try:
                # 1) Monitor existing position(s) and manage partial closes
                try:
                    self._monitor_and_manage_positions(dry_run=dry_run)
                except Exception as e:
                    logger.exception('Error in _monitor_and_manage_positions: %s', e)

                # 2) If in cooldown, skip signal generation
                now = time.time()
                if now < self.in_cooldown_until:
                    logger.debug('In cooldown until %s; skipping signal eval', self.in_cooldown_until)
                else:
                    # 3) Evaluate new signal only if no open position or position closed
                    #    We allow only one aggregated open position at a time (any direction).
                    positions = []
                    if mt5 is not None and self.mt5.connected:
                        positions = mt5.positions_get(symbol=self.cfg.SYMBOL) or []

                    if positions and len(positions) > 0:
                        # position(s) exist: rely on monitor, do not open new
                        logger.debug('Existing positions present (%d), skipping new-entry evaluation', len(positions))
                    else:
                        # No open position -> evaluate ML signal
                        df_now = None
                        try:
                            df_now = self.mt5.fetch_history_m5(300) if (mt5 is not None and self.mt5.connected) else None
                        except Exception as e:
                            logger.warning('Failed to fetch history for signal evaluation: %s', e)
                            df_now = None

                        if df_now is None or len(df_now) < max(50, self.cfg.LOOKBACK_BARS + 10):
                            logger.debug('Insufficient data for features; skipping')
                        else:
                            df_proc, Xs_now = self.fe.build_features(df_now, fit_scaler=False)
                            seq_tail = Xs_now[-self.cfg.LOOKBACK_BARS:] if Xs_now.shape[0] >= self.cfg.LOOKBACK_BARS else Xs_now
                            multi_probs = self.models.predict_proba_multi(Xs_now, seq_tail=seq_tail)
                            # compute strategy scores similar to main code
                            strategy_scores = {}
                            for s in self.cfg.STRATEGIES:
                                p = float(multi_probs.get(s, np.full(Xs_now.shape[0], 0.5))[-1])
                                try:
                                    reward_pred = float(self.models.predict_reward(Xs_now[-1:].astype(np.float32), s)[0])
                                except Exception:
                                    reward_pred = 0.0
                                reward_clamped = max(-1.0, min(reward_pred, 3.0))
                                score = p * max(0.0, reward_clamped)
                                strategy_scores[s] = {'prob': p, 'reward': reward_pred, 'score': score}

                            # choose best strategy
                            best_strat = max(strategy_scores.keys(), key=lambda k: strategy_scores[k]['score'])
                            best = strategy_scores[best_strat]
                            chosen_prob = best['prob']
                            logger.info('Strategy scores=%s chosen=%s prob=%.3f reward=%.3f score=%.4f',
                                        {k: (v['prob'], round(v['reward'],3), round(v['score'],3)) for k,v in strategy_scores.items()},
                                        best_strat, best['prob'], best['reward'], best['score'])

                            # Decide side with label confirmation (if available)
                            labels_multi, _ = construct_labels_and_rewards(df_proc, self.cfg)
                            comp = composite_label_from_multi(labels_multi, self.cfg)
                            label_idx = - (self.cfg.PRED_HORIZON + 1)
                            possible_label = None
                            if len(comp) >= abs(label_idx):
                                possible_label = int(comp[label_idx])
                            # require match between label and prob OR extremely strong prob-only
                            side_to_open = None
                            if possible_label == 1 and chosen_prob >= self.cfg.MIN_TRADE_PROB:
                                side_to_open = 'BUY'
                            elif possible_label == 0 and chosen_prob <= (1.0 - self.cfg.MIN_TRADE_PROB):
                                side_to_open = 'SELL'
                            else:
                                # allow prob-only if very strong
                                if chosen_prob >= self.cfg.PROB_ONLY_THRESHOLD:
                                    side_to_open = 'BUY'
                                elif chosen_prob <= (1.0 - self.cfg.PROB_ONLY_THRESHOLD):
                                    side_to_open = 'SELL'

                            if side_to_open is not None:
                                # Avoid repeat firing in short interval
                                if self.last_side == side_to_open and (now - self.last_trade_time) < self.MIN_SIGNAL_INTERVAL:
                                    logger.info('Signal %s but MIN_SIGNAL_INTERVAL (%.1fs) not elapsed since last trade, skipping', side_to_open, self.MIN_SIGNAL_INTERVAL)
                                else:
                                    # Compute volume (single aggregated position)
                                    price_tick = mt5.symbol_info_tick(self.cfg.SYMBOL) if (mt5 is not None and self.mt5.connected) else None
                                    exec_price = float(price_tick.ask if side_to_open == 'BUY' else price_tick.bid) if price_tick else float(df_proc['close'].iloc[-1])
                                    # derive ATR
                                    atr_val = float(df_proc['ATR'].iloc[-1]) if 'ATR' in df_proc.columns else max(1e-3, exec_price * 0.001)
                                    # compute target lot via risk manager (we'll scale allowed to MAX_VOLUME)
                                    account_info = mt5.account_info() if (mt5 is not None and self.mt5.connected) else None
                                    account_balance = float(account_info.balance) if account_info is not None else 10000.0
                                    # set an aggressive single trade risk budget (e.g., 2x RISK_PER_TRADE)
                                    single_risk = min(0.05, self.cfg.RISK_PER_TRADE * 4)  # e.g. allow bigger single position but capped
                                    # temporarily override risk manager per this allocation
                                    # compute a desired volume using risk_amount = account_balance * single_risk
                                    desired_volume = self._compute_volume_for_risk(account_balance, exec_price, atr_val, single_risk)
                                    # enforce symbol constraints
                                    sym = mt5.symbol_info(self.cfg.SYMBOL) if (mt5 is not None and self.mt5.connected) else None
                                    vol_min = float(getattr(sym, 'volume_min', self.MIN_VOLUME_FOR_TRADE))
                                    vol_step = float(getattr(sym, 'volume_step', 0.01))
                                    vol_max = float(getattr(sym, 'volume_max', self.cfg.MAX_VOLUME))
                                    desired_volume = max(vol_min, min(desired_volume, vol_max))
                                    # round to step
                                    try:
                                        steps = round((desired_volume - vol_min) / vol_step)
                                        desired_volume = float(vol_min + steps * vol_step)
                                    except Exception:
                                        desired_volume = float(max(vol_min, min(desired_volume, vol_max)))

                                    logger.info('Prepared aggregated %s open: price=%.5f atr=%.5f desired_vol=%.3f', side_to_open, exec_price, atr_val, desired_volume)

                                    if dry_run:
                                        logger.info('Dry-run: WOULD OPEN aggregated %s volume=%.3f at price=%.5f', side_to_open, desired_volume, exec_price)
                                        # emulate that it would be opened and then let monitor handle closing (only for dry-run)
                                        self.last_side = side_to_open
                                        self.last_trade_time = now
                                        # set a pseudo open position record for dry-run
                                        self.open_position_volume = desired_volume
                                        self.open_position_type = side_to_open
                                        self.partial_close_state = [{'frac': pc['frac'], 'executed': False, 'tp': None} for pc in self.PARTIAL_CLOSES]
                                    else:
                                        # open aggregated order
                                        try:
                                            res = self._open_aggregated_position(side_to_open, desired_volume, exec_price, atr_val)
                                            if res is not None:
                                                self.last_side = side_to_open
                                                self.last_trade_time = now
                                        except Exception as e:
                                            logger.exception('Failed to open aggregated position: %s', e)
            except Exception as e:
                logger.exception('Unhandled error in live loop: %s', e)

            # enforce loop delay
            elapsed = time.time() - loop_start
            if elapsed < loop_delay:
                time.sleep(max(0, loop_delay - elapsed))

        # end loop
        if self.mt5.connected:
            self.mt5.disconnect()
        logger.info('ExecutorLive stopped')

    # ---------- helper: compute volume for specified risk fraction ----------
    def _compute_volume_for_risk(self, account_balance: float, price: float, atr: float, risk_fraction: float) -> float:
        """
        Conservative volume calculation:
         - risk_amount = account_balance * risk_fraction
         - sl_distance = atr (we size SL at ~ATR)
         - use MT5 symbol contract_size * point to compute pip value if available
        """
        risk_amount = account_balance * risk_fraction
        sl_distance = max(1e-6, atr)
        try:
            sym = mt5.symbol_info(self.cfg.SYMBOL)
            contract_size = float(getattr(sym, 'trade_contract_size', 100.0))
            point = float(getattr(sym, 'point', 0.01))
            pip_value_per_lot = contract_size * point
            vol = risk_amount / (sl_distance * pip_value_per_lot)
        except Exception:
            vol = risk_amount / (sl_distance * 1000.0)
        vol = max(0.01, min(vol, self.cfg.MAX_VOLUME))
        return float(vol)

    # ---------- open aggregated position ----------
    def _open_aggregated_position(self, side: str, volume: float, price: float, atr_val: float):
        """
        Send a single market order to open aggregated position.
        We set SL/TP conservatively (SL = ATR * 1.2, TP partials per PARTIAL_CLOSES).
        """
        if mt5 is None:
            raise RuntimeError('MT5 not available')

        sym = mt5.symbol_info(self.cfg.SYMBOL)
        if sym is None:
            logger.warning('Symbol info not available')
            return None

        digits = int(getattr(sym, 'digits', 2))
        point = float(getattr(sym, 'point', 0.01))
        min_stop_level = int(getattr(sym, 'trade_stops_level', 0))
        min_stop_distance = max(min_stop_level * point, point)

        # protective SL chosen as 1.2 * ATR
        sl_dist = max(atr_val * 1.2, min_stop_distance * 2)
        if side == 'BUY':
            price_exec = float(round(price, digits))
            sl_price = float(round(price_exec - sl_dist, digits))
            tp_price = float(round(price_exec + atr_val * self.PARTIAL_CLOSES[0]['tp_atr'], digits))
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price_exec = float(round(price, digits))
            sl_price = float(round(price_exec + sl_dist, digits))
            tp_price = float(round(price_exec - atr_val * self.PARTIAL_CLOSES[0]['tp_atr'], digits))
            order_type = mt5.ORDER_TYPE_SELL

        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': self.cfg.SYMBOL,
            'volume': float(volume),
            'type': order_type,
            'price': price_exec,
            'sl': float(sl_price),
            'tp': float(tp_price),
            'deviation': 20,
            'magic': 555001,
            'comment': 'walk_ai_aggregated_open',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
        }

        logger.info('Sending aggregated open request: %s', request)
        result = mt5.order_send(request)
        logger.info('Order send result: %s', result)
        # If success, record the open position details
        try:
            if hasattr(result, 'retcode') and result.retcode == 10009:
                # success: find position(s)
                time.sleep(0.3)
                positions = mt5.positions_get(symbol=self.cfg.SYMBOL) or []
                # find the new position by matching side and volume closish
                for p in positions:
                    p_side = 'BUY' if int(p.type) == mt5.ORDER_TYPE_BUY else 'SELL'
                    # use some tolerance for volume equality
                    if p_side == side and abs(p.volume - volume) < max(0.0001, volume * 0.2):
                        self.open_position_volume = float(p.volume)
                        self.open_position_type = p_side
                        self.open_position_ticket = int(getattr(p, 'ticket', 0))
                        # reset partial close state
                        self.partial_close_state = [{'frac': pc['frac'], 'executed': False, 'tp': None, 'tp_price': None} for pc in self.PARTIAL_CLOSES]
                        logger.info('Recorded new aggregated position ticket=%s type=%s vol=%.3f', self.open_position_ticket, self.open_position_type, self.open_position_volume)
                        return result
                # fallback: if not found, still record some info
                self.open_position_volume = volume
                self.open_position_type = side
                self.open_position_ticket = getattr(result, 'order', None) or None
                self.partial_close_state = [{'frac': pc['frac'], 'executed': False, 'tp': None, 'tp_price': None} for pc in self.PARTIAL_CLOSES]
                return result
            else:
                logger.warning('Aggregated open returned non-success retcode: %s', getattr(result, 'retcode', 'unknown'))
                return result
        except Exception as e:
            logger.exception('Post-order handling failed: %s', e)
            return result

    # ---------- monitoring & partial close logic ----------
    def _monitor_and_manage_positions(self, dry_run: bool = True):
        """
        Monitor positions for the symbol and execute partial closes based on configured TP levels.
        Also detect full close and set cooldown accordingly.
        """
        # if no position recorded, refresh and return
        positions = []
        if mt5 is not None and self.mt5.connected:
            positions = mt5.positions_get(symbol=self.cfg.SYMBOL) or []

        if not positions and self.open_position_volume <= 0:
            # nothing open
            return

        # compute market regime to adapt cooldown later
        df_hist = None
        try:
            df_hist = self.mt5.fetch_history_m5(300) if (mt5 is not None and self.mt5.connected) else None
        except Exception:
            df_hist = None

        regime = self._detect_market_regime(df_hist) if df_hist is not None else 'chop'
        # if positions present, manage them
        if positions and len(positions) > 0:
            # For simplicity, aggregate first position (should be only one)
            # Use the latest tick to check current price
            tick = mt5.symbol_info_tick(self.cfg.SYMBOL)
            bid = float(tick.bid) if tick else None
            ask = float(tick.ask) if tick else None
            mid = 0.5 * (bid + ask) if (bid is not None and ask is not None) else None

            # compute ATR from recent ticks if available for TP sizing
            atr_val = None
            try:
                if df_hist is not None and 'ATR' in df_hist.columns:
                    atr_val = float(df_hist['ATR'].iloc[-1])
                elif df_hist is not None:
                    atr_val = float(compute_atr(df_hist['high'], df_hist['low'], df_hist['close'], max(5, self.cfg.ATR_PERIOD)).iloc[-1])
            except Exception:
                atr_val = None
            if atr_val is None:
                atr_val = max(1e-3, (ask or mid or 1.0) * 0.001)

            # Manage each open position: try to partial-close based on configured steps
            total_open_volume = sum([float(p.volume) for p in positions])
            # store the aggregate metrics
            self.open_position_volume = total_open_volume
            p = positions[0]
            pos_side = 'BUY' if int(p.type) == mt5.ORDER_TYPE_BUY else 'SELL'
            self.open_position_type = pos_side
            self.open_position_ticket = int(getattr(p, 'ticket', 0))
            logger.debug('Monitoring existing pos ticket=%s side=%s vol=%.3f', self.open_position_ticket, pos_side, self.open_position_volume)

            # Check partial close triggers: for each configured step that is not executed, compute TP price
            for idx, pc in enumerate(self.partial_close_state):
                if pc.get('executed', False):
                    continue
                frac = float(pc.get('frac', 0.0))
                tier_atr = float(self.PARTIAL_CLOSES[idx]['tp_atr'])
                # compute target depending on side
                if pos_side == 'BUY':
                    tp_price = (ask if ask is not None else mid) + tier_atr * atr_val
                else:
                    tp_price = (bid if bid is not None else mid) - tier_atr * atr_val
                # record tp price for logging
                pc['tp_price'] = tp_price

                # decide if reached (for safety use >= for BUY and <= for SELL)
                # use last close price if tick missing
                cur_price = mid if mid is not None else float(df_hist['close'].iloc[-1])
                reached = (pos_side == 'BUY' and cur_price >= tp_price) or (pos_side == 'SELL' and cur_price <= tp_price)

                if reached:
                    # determine volume to close (fraction of current remaining)
                    remaining_vol = self.open_position_volume
                    close_vol = max(self.MIN_VOLUME_FOR_TRADE, round(remaining_vol * frac, 3))
                    # cap close_vol not to exceed remaining
                    close_vol = min(close_vol, remaining_vol)
                    if close_vol < self.MIN_VOLUME_FOR_TRADE:
                        logger.info('Remaining volume too small to partial-close (remaining=%.4f)', remaining_vol)
                        pc['executed'] = True
                        continue

                    # build close request (opposite side order)
                    if pos_side == 'BUY':
                        close_type = mt5.ORDER_TYPE_SELL
                        close_price = bid if bid is not None else cur_price
                    else:
                        close_type = mt5.ORDER_TYPE_BUY
                        close_price = ask if ask is not None else cur_price

                    # round volume and price
                    sym = mt5.symbol_info(self.cfg.SYMBOL)
                    digits = int(getattr(sym, 'digits', 2))
                    close_price = float(round(close_price, digits))

                    logger.info('Partial close triggered: side=%s close_vol=%.3f tp_price=%.5f cur_price=%.5f', pos_side, close_vol, tp_price, cur_price)
                    if dry_run:
                        logger.info('Dry-run: WOULD partial-close %s vol=%.3f at price=%.5f', pos_side, close_vol, close_price)
                        # emulate partial close
                        self.open_position_volume = max(0.0, self.open_position_volume - close_vol)
                        pc['executed'] = True
                        continue

                    # prepare request
                    req = {
                        'action': mt5.TRADE_ACTION_DEAL,
                        'symbol': self.cfg.SYMBOL,
                        'volume': float(close_vol),
                        'type': close_type,
                        'price': close_price,
                        'deviation': 20,
                        'magic': 555001,
                        'comment': 'walk_ai_partial_close',
                        'type_time': mt5.ORDER_TIME_GTC,
                        'type_filling': mt5.ORDER_FILLING_IOC,
                    }
                    try:
                        r = mt5.order_send(req)
                        logger.info('Partial close order_send result: %s', r)
                        # if successful, mark executed and update internal remaining vol
                        if hasattr(r, 'retcode') and r.retcode == 10009:
                            # allow a brief wait and refresh positions
                            time.sleep(0.2)
                            positions = mt5.positions_get(symbol=self.cfg.SYMBOL) or []
                            self.open_position_volume = sum([float(p.volume) for p in positions]) if positions else 0.0
                            pc['executed'] = True
                        else:
                            logger.warning('Partial close returned non-success retcode: %s', getattr(r, 'retcode', None))
                            # still mark executed to avoid infinite loop; you may prefer retry logic
                            pc['executed'] = True
                    except Exception as e:
                        logger.exception('Exception executing partial close: %s', e)
                        # do not mark executed so we can try next loop; but avoid rapid retries
                        pc['executed'] = False
                        time.sleep(0.5)

            # After processing partials, if no volume remains -> full closed: set cooldown
            if self.open_position_volume <= 0.00001 or not positions:
                # all closed
                self.open_position_volume = 0.0
                self.open_position_ticket = None
                self.open_position_type = None
                self.partial_close_state = []
                # compute cooldown depending on regime
                multiplier = self.REGIME_MULTIPLIER.get(regime, 1.0)
                cooldown = self.COOLDOWN_BASE_SEC * multiplier
                self.in_cooldown_until = time.time() + cooldown
                self.last_side = None
                self.last_trade_time = time.time()
                logger.info('Position fully closed. Entering cooldown for %.1fs (regime=%s multiplier=%.2f)', cooldown, regime, multiplier)
        else:
            # We previously thought position existed but MT5 shows none -> clear local record and set cooldown
            if self.open_position_volume > 0:
                self.open_position_volume = 0.0
                self.open_position_ticket = None
                self.open_position_type = None
                self.partial_close_state = []
                multiplier = self.REGIME_MULTIPLIER.get(regime, 1.0)
                cooldown = self.COOLDOWN_BASE_SEC * multiplier
                self.in_cooldown_until = time.time() + cooldown
                self.last_side = None
                self.last_trade_time = time.time()
                logger.info('Detected remote position close. Entering cooldown %.1fs (regime=%s)', cooldown, regime)

    # ---------- market regime detection ----------
    def _detect_market_regime(self, df_hist: pd.DataFrame) -> str:
        """
        Quick heuristic regime detector:
          - compute Hurst on last N closes: hurst > 0.6 => trending
          - else if ATR relatively small vs rolling mean => chop / squeeze
          - else meanrev if variance high but hurst low
        Returns one of {'trend', 'meanrev', 'chop'}
        """
        try:
            close = df_hist['close'].dropna().values[-500:]
            hurst_val = hurst_exponent(close) if len(close) >= 50 else 0.5
            atr = compute_atr(df_hist['high'], df_hist['low'], df_hist['close'], max(5, self.cfg.ATR_PERIOD)).iloc[-1]
            atr_rel = atr / (df_hist['close'].rolling(200).mean().iloc[-1] if df_hist['close'].rolling(200).mean().iloc[-1] > 0 else 1.0)
            # rules:
            if hurst_val >= 0.60:
                return 'trend'
            if atr_rel < 0.0004:  # very low ATR relative -> squeeze / chop
                return 'chop'
            # otherwise mean-reversion environment
            return 'meanrev'
        except Exception:
            return 'chop'

    def stop(self):
        self._stop = True


# ---------- CLI / Training / Backtesting flow ----------
def compute_performance_metrics(pnl_series: np.ndarray) -> Dict[str, float]:
    if pnl_series is None or len(pnl_series) == 0:
        return {'sharpe': 0.0, 'sortino': 0.0, 'max_drawdown': 0.0}
    returns = np.asarray(pnl_series)
    if returns.std() == 0:
        return {'sharpe': 0.0, 'sortino': 0.0, 'max_drawdown': 0.0}
    sharpe = np.mean(returns) / (returns.std() + 1e-12) * math.sqrt(252)
    neg = returns[returns < 0]
    sortino = np.mean(returns) / (neg.std() + 1e-12) if neg.size > 0 else float('inf')
    cum = np.cumsum(returns)
    drawdown = np.maximum.accumulate(cum) - cum
    max_dd = drawdown.max() if len(drawdown) > 0 else 0.0
    return {'sharpe': float(sharpe), 'sortino': float(sortino), 'max_drawdown': float(max_dd)}

def walk_forward_train_and_backtest(cfg: Config, hist_df: pd.DataFrame):
    fe = FeatureEngineer(cfg)
    df_proc, Xs = fe.build_features(hist_df, fit_scaler=True)
    joblib.dump(fe.scaler, os.path.join(cfg.MODEL_DIR, 'fe_scaler.pkl'))
    labels_multi, rewards_multi = construct_labels_and_rewards(df_proc, cfg)
    comp_labels = composite_label_from_multi(labels_multi, cfg)

    n = len(Xs)
    fold_size = max(1, int(n * 0.1))
    min_train = max(200, int(n * 0.2))
    oos_probs = {s: np.full(n, np.nan) for s in cfg.STRATEGIES}
    oos_rewards = {s: np.full(n, np.nan) for s in cfg.STRATEGIES}

    mm_total = ModelManager(cfg)

    start = min_train
    while start + fold_size <= n:
        train_idx = np.arange(0, start)
        test_idx = np.arange(start, start + fold_size)
        logger.info("WALK-FWD: train=%d..%d test=%d..%d", train_idx[0], train_idx[-1], test_idx[0], test_idx[-1])

        X_train = Xs[train_idx]
        X_test = Xs[test_idx]

        for strat in cfg.STRATEGIES:
            y = labels_multi[strat]
            r = rewards_multi[strat]
            mask_train = (np.arange(len(y)) < start) & (y != -1)
            if mask_train.sum() < 80:
                logger.info("WALK-FWD skip strat=%s due to small train N=%d", strat, mask_train.sum())
                continue
            Xf_train = Xs[mask_train]
            yf_train = y[mask_train]
            rf_train = r[mask_train]

            try:
                mm_total.train_tree(Xf_train, yf_train, strat)
            except Exception as e:
                logger.exception("train_tree fold failed %s: %s", strat, e)
            try:
                mm_total.train_reward_regressor(Xf_train, rf_train, strat)
            except Exception as e:
                logger.exception("train_reward_regressor failed %s: %s", strat, e)
            try:
                mm_total.train_lstm(Xf_train, yf_train, seq_len=cfg.LOOKBACK_BARS, strat=strat)
            except Exception as e:
                logger.exception("train_lstm failed %s: %s", strat, e)

            try:
                p_test = mm_total.predict_proba_multi(X_test, seq_tail=Xs[test_idx[0]:test_idx[0]+cfg.LOOKBACK_BARS] if Xs.shape[0] > cfg.LOOKBACK_BARS else Xs[test_idx])
                oos_probs[strat][test_idx] = p_test[strat]
            except Exception as e:
                logger.exception("predict_proba_multi OOS failed %s: %s", strat, e)

            try:
                r_test = mm_total.predict_reward(X_test, strat)
                oos_rewards[strat][test_idx] = r_test
            except Exception as e:
                logger.exception("predict_reward OOS failed %s: %s", strat, e)

        start += fold_size

    probs_stack = np.vstack([np.nan_to_num(oos_probs[s], nan=0.5) for s in cfg.STRATEGIES])
    weights = np.array([cfg.STRATEGY_WEIGHTS[s] for s in cfg.STRATEGIES])
    meta_oos = np.average(probs_stack, axis=0, weights=weights)

    bt = Backtester(cfg)
    sim = bt.simulate(df_proc, meta_oos, comp_labels)
    metrics = compute_performance_metrics(sim.get('pnl_series', []))
    logger.info("WALK-FWD backtest results: trades=%d win_rate=%.3f net=%.5f metrics=%s",
                sim['trades'], sim['win_rate'], sim['net'], metrics)
    mm_total.save()
    return mm_total, fe, df_proc, meta_oos, comp_labels, sim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run self-test on synthetic data')
    parser.add_argument('--backtest', action='store_true', help='Run backtest using MT5 history')
    parser.add_argument('--live', action='store_true', help='Start live executor (requires MT5)')
    parser.add_argument('--dry', action='store_true', help='Live dry-run (no real orders)')
    args = parser.parse_args()

    cfg = Config()
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    logger.info('Starting bot with device=%s, seed=%d', cfg.DEVICE, cfg.SEED)

    if args.test:
        logger.info('Running synthetic self-test')
        dt = pd.date_range(end=pd.Timestamp.utcnow(), periods=2000, freq='5min')
        rng = np.random.RandomState(cfg.SEED)
        price = 2000.0
        rows = []
        for t in dt:
            change = rng.normal(loc=0.0, scale=0.001)
            new_price = price * (1 + change)
            high = max(price, new_price) * (1 + abs(rng.normal(0, 0.0001)))
            low = min(price, new_price) * (1 - abs(rng.normal(0, 0.0001)))
            rows.append({'time': t, 'open': price, 'high': high, 'low': low, 'close': new_price, 'tick_volume': int(abs(rng.normal(100, 20)))})
            price = new_price
        df = pd.DataFrame(rows)
        mm, fe, df_proc, probs, labels, sim = walk_forward_train_and_backtest(cfg, df)
        logger.info('Synthetic test complete. Sim results: %s', sim)
        return

    if args.backtest:
        if mt5 is None:
            logger.error('MT5 not available for historical backtest in this run')
            return
        mt5c = MT5Connector(cfg)
        if not mt5c.connect():
            logger.error('MT5 connect failed for backtest')
            return
        df_hist = mt5c.fetch_history_m5(cfg.HISTORY_BARS)
        mm, fe, df_proc, probs, labels, sim = walk_forward_train_and_backtest(cfg, df_hist)
        logger.info('Backtest done: %s', sim)
        mt5c.disconnect()
        return

    if args.live:
        mm = ModelManager.load(cfg)
        fe = FeatureEngineer(cfg)
        fe_scaler_path = os.path.join(cfg.MODEL_DIR, 'fe_scaler.pkl')
        if os.path.exists(fe_scaler_path):
            try:
                fe.scaler = joblib.load(fe_scaler_path)
                logger.info('Loaded feature scaler for live from %s', fe_scaler_path)
            except Exception as e:
                logger.warning('Failed to load fe_scaler for live: %s', e)
        execer = ExecutorLive(cfg, mm, fe)
        execer.start(dry_run=args.dry)
        return

    parser.print_help()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception('Unhandled exception in main: %s', e)
        sys.exit(1)
