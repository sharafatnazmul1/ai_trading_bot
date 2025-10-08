#!/usr/bin/env python3
"""
Patched AI trading bot (full file)
Includes:
 - multi-timeframe fetch (M5/M15/H1)
 - feature engineering (EMA, BB width, zscore, vol_norm, dir_blend)
 - saved scaler & model loading (joblib)
 - dynamic TP from ML prediction, ATR-based SL
 - reinforcement feedback from closed deals (recent pnls adjust sensitivity)
 - safety: HOLD blocks order placing; live loop with dry-run option
"""
import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import joblib

# sklearn
from sklearn.preprocessing import MinMaxScaler

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# MT5 optional
try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

# ---------- Logging ----------
def setup_logging(logfile='logs/ai_trading_bot.log'):
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    root = logging.getLogger()
    if root.handlers:
        root.handlers.clear()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    fh = RotatingFileHandler(logfile, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(ch)
    root.addHandler(fh)

setup_logging()
logger = logging.getLogger("ai_trading_bot")

# ---------- Config ----------
@dataclass
class Config:
    SYMBOL: str = "XAUUSDm"
    HISTORY_BARS: int = 100000
    ATR_PERIOD: int = 14
    RSI_PERIOD: int = 14
    MA_PERIOD: int = 50
    PRICE_THRESHOLD_PCT: float = 0.0005
    PRICE_THRESHOLD_ATR_MULTIPLIER: float = 0.8
    LOOKBACK: int = 12
    LOT_SIZE: float = 0.01
    MAX_LOT: float = 0.1
    DEVICE: str = "cpu"
    MODEL_DIR: str = "models"
    EMA_SHORT: int = 5
    EMA_LONG: int = 20
    LAG: int = 5
    LIVE_LOOP_SLEEP: int = 60

# ---------- Indicators ----------
def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    return atr

def compute_rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window).mean()
    rs = avg_gain / (avg_loss.replace(0, 1e-12))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def compute_bband_width(close: pd.Series, window: int) -> pd.Series:
    ma = close.rolling(window=window, min_periods=1).mean()
    std = close.rolling(window=window, min_periods=1).std().fillna(0.0)
    width = (2 * std) / ma.replace(0, np.nan)
    return width.fillna(0.0)

# ---------- DataFetcher ----------
class DataFetcher:
    def __init__(self, config: Config):
        self.config = config
        self.scaler: Optional[MinMaxScaler] = None
        self.mode_scaler: Optional[MinMaxScaler] = None
        self.feature_names: Optional[list] = None
        self.last_df: Optional[pd.DataFrame] = None
        self.raw_close: Optional[np.ndarray] = None
        self.recent_pnls: list = []
        self.processed_deals: set = set()
        self.pending_orders: dict = {}
        self.sensitivity: float = 1.0

    def connect_mt5(self) -> bool:
        if mt5 is None:
            logger.warning("MetaTrader5 package not available. Live trading disabled.")
            return False
        try:
            ok = mt5.initialize()
            if not ok:
                logger.warning("MT5 initialize returned False")
                return False
            logger.info("Connected to MT5 successfully")
            return True
        except Exception as e:
            logger.exception("Exception connecting to MT5: %s", e)
            return False

    def fetch_rates(self, bars: int) -> pd.DataFrame:
        if mt5 is None:
            raise RuntimeError("MT5 not available in this environment.")
        symbol = self.config.SYMBOL
        timeframe = mt5.TIMEFRAME_M5 if hasattr(mt5, 'TIMEFRAME_M5') else 5
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError("Failed to fetch rates from MT5 or zero bars returned")
        df = pd.DataFrame(rates)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def fetch_multi_timeframe(self, bars_m5: int = 500) -> pd.DataFrame:
        if mt5 is None:
            raise RuntimeError('MT5 not available for multi-timeframe fetch')
        TF5 = getattr(mt5, 'TIMEFRAME_M5', 5)
        TF15 = getattr(mt5, 'TIMEFRAME_M15', 15)
        TFH1 = getattr(mt5, 'TIMEFRAME_H1', 60)
        rates5 = mt5.copy_rates_from_pos(self.config.SYMBOL, TF5, 0, bars_m5)
        if rates5 is None or len(rates5) == 0:
            raise RuntimeError('Failed to fetch M5 rates')
        df5 = pd.DataFrame(rates5)
        df5['time'] = pd.to_datetime(df5['time'], unit='s')
        bars15 = max(10, bars_m5 // 3)
        barsh1 = max(10, bars_m5 // 12)
        rates15 = mt5.copy_rates_from_pos(self.config.SYMBOL, TF15, 0, bars15) 
        ratesh1 = mt5.copy_rates_from_pos(self.config.SYMBOL, TFH1, 0, barsh1) 
        if len(rates15) == 0 or len(ratesh1) == 0:
            df5 = df5.reset_index(drop=True)
            df5['M15_dir'] = 0.0
            df5['H1_dir'] = 0.0
            df5['dir_blend'] = 0.0
            return df5
        df15 = pd.DataFrame(rates15); df15['time'] = pd.to_datetime(df15['time'], unit='s')
        dfh1 = pd.DataFrame(ratesh1); dfh1['time'] = pd.to_datetime(dfh1['time'], unit='s')
        s15 = pd.Series(df15['close'].values, index=pd.DatetimeIndex(df15['time']))
        sh1 = pd.Series(dfh1['close'].values, index=pd.DatetimeIndex(dfh1['time']))
        idx5 = pd.DatetimeIndex(df5['time'])
        s15_ff = s15.reindex(idx5, method='ffill')
        sh1_ff = sh1.reindex(idx5, method='ffill')
        m15_dir = s15_ff.diff().fillna(0.0).apply(np.sign)
        h1_dir = sh1_ff.diff().fillna(0.0).apply(np.sign)
        df5 = df5.reset_index(drop=True)
        df5['M15_dir'] = m15_dir.values
        df5['H1_dir'] = h1_dir.values
        df5['dir_blend'] = ((df5['M15_dir'].fillna(0.0) + df5['H1_dir'].fillna(0.0)) / 2.0).fillna(0.0)
        return df5

    def preprocess_data(self, df: pd.DataFrame, fit_scaler: bool = True) -> np.ndarray:
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'])
        df['ATR'] = compute_atr(df['high'], df['low'], df['close'], self.config.ATR_PERIOD)
        df['RSI'] = compute_rsi(df['close'], self.config.RSI_PERIOD)
        df['MA'] = compute_sma(df['close'], self.config.MA_PERIOD)
        df['log_return'] = np.log(df['close']).diff().fillna(0.0)
        df['EMA5'] = compute_ema(df['close'], span=self.config.EMA_SHORT)
        df['EMA20'] = compute_ema(df['close'], span=self.config.EMA_LONG)
        df['bb_width'] = compute_bband_width(df['close'], window=self.config.EMA_LONG)
        df['hour'] = df['time'].dt.hour + df['time'].dt.minute / 60.0
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

        # lag features
        for lag in range(1, self.config.LAG + 1):
            df[f'lag_{lag}'] = df['log_return'].shift(lag).fillna(0.0)

        # z-score and volatility normalized features
        mean = df['log_return'].rolling(200).mean()
        std = df['log_return'].rolling(200).std().replace(0, np.nan)
        df['zscore_ret'] = ((df['log_return'] - mean) / std).fillna(0.0)
        df['vol_norm'] = df['ATR'] / df['ATR'].rolling(200).mean().fillna(df['ATR'].mean())

        # --- NEW micro / scalping features (helps capture candle micro-structure) ---
        # body ratio: size of body relative to full range (spike vs wick)
        df['body_ratio'] = ((df['close'] - df['open']).abs() / ((df['high'] - df['low']) + 1e-9)).fillna(0.0)
        # short directional memory (sum of last 3 candles' direction)
        df['dir_memory'] = np.sign(df['close'] - df['open']).rolling(3, min_periods=1).sum().fillna(0.0)
        # short momentum (3-bar average of log returns)
        df['momentum'] = df['log_return'].rolling(3, min_periods=1).mean().fillna(0.0)
        # ATR deviation: current ATR relative to 50-bar ATR mean (volatility spike detector)
        df['atr_50_mean'] = df['ATR'].rolling(50, min_periods=1).mean()
        df['atr_50_std'] = df['ATR'].rolling(50, min_periods=1).std().fillna(1e-9)
        df['atr_dev'] = ((df['ATR'] - df['atr_50_mean']) / (df['atr_50_std'] + 1e-9)).fillna(0.0)
        # drop helper cols later (atr_50_mean/std), keep atr_dev only
        df = df.drop(columns=['atr_50_mean','atr_50_std'], errors='ignore')

        if 'dir_blend' not in df.columns:
            df['dir_blend'] = 0.0

        # Prepare final frame and drop NaNs
        df = df.dropna().reset_index(drop=True)
        if df.empty:
            raise ValueError('Preprocessing produced empty dataframe after feature engineering')

        self.last_df = df.copy()
        self.raw_close = df['close'].values

        # Feature list (order matters) — include new micro features
        feature_cols = [
            'open','high','low','close','ATR','RSI','MA','log_return',
            'EMA5','EMA20','bb_width','hour_sin','hour_cos','zscore_ret','vol_norm','dir_blend',
            'body_ratio','dir_memory','momentum','atr_dev'
        ] + [f'lag_{i}' for i in range(1, self.config.LAG + 1)]

        self.feature_names = feature_cols
        features_df = df[feature_cols].ffill()
        if fit_scaler:
            self.scaler = MinMaxScaler()
            scaled = self.scaler.fit_transform(features_df.values)
        else:
            if self.scaler is None:
                raise RuntimeError('Scaler missing: cannot preprocess in inference mode')
            scaled = self.scaler.transform(features_df.values)
        return scaled.astype(np.float32)


    def prepare_mode_data(self, df: pd.DataFrame, fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        df = df.copy()
        df['ATR'] = compute_atr(df['high'], df['low'], df['close'], self.config.ATR_PERIOD)
        df['RSI'] = compute_rsi(df['close'], self.config.RSI_PERIOD)
        df['MA'] = compute_sma(df['close'], self.config.MA_PERIOD)
        df = df.dropna().reset_index(drop=True)
        if len(df) < 2:
            return np.zeros((0,3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        labels = []
        mult = getattr(self.config, 'PRICE_THRESHOLD_ATR_MULTIPLIER', 0.8)
        for i in range(1, len(df)):
            prev = df['close'].iloc[i-1]; curr = df['close'].iloc[i]
            move = abs(curr - prev); threshold = df['ATR'].iloc[i] * mult
            labels.append(1.0 if move > threshold * 0.6 else 0.0)
        feat = df[['ATR','RSI','MA']].iloc[1:].values
        if fit_scaler:
            self.mode_scaler = MinMaxScaler()
            scaled_features = self.mode_scaler.fit_transform(feat)
        else:
            if getattr(self, 'mode_scaler', None) is None:
                raise RuntimeError('Mode scaler missing for inference')
            scaled_features = self.mode_scaler.transform(feat)
        labels_arr = np.array(labels, dtype=np.float32)
        self.last_mode_labels = labels_arr
        logger.info(f"Mode labels balance: {np.mean(labels_arr) * 100:.2f}% aggressive")
        return scaled_features.astype(np.float32), labels_arr

# ---------- Datasets ----------
class PriceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

class ModeDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
        # return scalar/1-d target (DataLoader will batch into shape (batch,))
        return x_tensor, y_tensor
 
#        if y_tensor.dim() == 0:
#            y_tensor = y_tensor.unsqueeze(0)
#        elif y_tensor.dim() > 1:
#            y_tensor = y_tensor.view(-1)
#        if y_tensor.dim() == 1:
#            y_tensor = y_tensor.unsqueeze(1)
#        return x_tensor, y_tensor

# ---------- Models ----------
class LSTMModel(nn.Module):
    """
    Bidirectional LSTM with simple attention pooling for scalping signals.
    Returns a scalar prediction (batch,).
    """
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        # attention layer: maps hidden*2 -> 1 (score)
        self.attn = nn.Linear(hidden_size * 2, 1)
        # final MLP
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)                # out: (B, T, hidden*2)
        # attention weights across time axis
        scores = self.attn(out).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        context = torch.sum(weights * out, dim=1)  # (B, hidden*2)
        out = self.fc(context)                # (B, 1)
        return out.squeeze(-1)               # (B,)


class ModeClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)   # returns shape (batch,)


# ---------- Trainer ----------
class ModelTrainer:
    def __init__(self, config: Config, data_fetcher: DataFetcher):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        self.data_fetcher = data_fetcher
        self.price_model: Optional[LSTMModel] = None
        self.mode_model: Optional[ModeClassifier] = None
        self.price_optimizer = None
        self.mode_optimizer = None
        self.price_criterion = nn.MSELoss()
        self.mode_criterion = nn.BCEWithLogitsLoss()
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)

    def save_model(self):
        try:
            os.makedirs(self.config.MODEL_DIR, exist_ok=True)
            price_path = os.path.join(self.config.MODEL_DIR, 'price_model.pt')
            mode_path = os.path.join(self.config.MODEL_DIR, 'mode_model.pt')
            scaler_path = os.path.join(self.config.MODEL_DIR, 'scaler.pkl')
            mode_scaler_path = os.path.join(self.config.MODEL_DIR, 'mode_scaler.pkl')
            fn_path = os.path.join(self.config.MODEL_DIR, 'feature_names.pkl')
            if self.price_model is not None:
                torch.save(self.price_model.state_dict(), price_path)
            if self.mode_model is not None:
                torch.save(self.mode_model.state_dict(), mode_path)
            if getattr(self.data_fetcher, 'scaler', None) is not None:
                joblib.dump(self.data_fetcher.scaler, scaler_path)
            if getattr(self.data_fetcher, 'mode_scaler', None) is not None:
                joblib.dump(self.data_fetcher.mode_scaler, mode_scaler_path)
            if getattr(self.data_fetcher, 'feature_names', None) is not None:
                joblib.dump(self.data_fetcher.feature_names, fn_path)
            logger.info('Models and scalers saved')
        except Exception as e:
            logger.exception('Failed to save models: %s', e)

    def try_load_models(self) -> bool:
        try:
            price_path = os.path.join(self.config.MODEL_DIR, 'price_model.pt')
            mode_path = os.path.join(self.config.MODEL_DIR, 'mode_model.pt')
            scaler_path = os.path.join(self.config.MODEL_DIR, 'scaler.pkl')
            mode_scaler_path = os.path.join(self.config.MODEL_DIR, 'mode_scaler.pkl')
            fn_path = os.path.join(self.config.MODEL_DIR, 'feature_names.pkl')
            if not (os.path.exists(price_path) and os.path.exists(mode_path) and os.path.exists(scaler_path)):
                logger.info('Saved models/scaler not found; will train new models.')
                return False
            self.data_fetcher.scaler = joblib.load(scaler_path)
            if os.path.exists(mode_scaler_path):
                self.data_fetcher.mode_scaler = joblib.load(mode_scaler_path)
            if os.path.exists(fn_path):
                self.data_fetcher.feature_names = joblib.load(fn_path)
            input_size = len(self.data_fetcher.feature_names) if self.data_fetcher.feature_names else 15
            self.price_model = LSTMModel(input_size=input_size).to(self.device)
            self.price_model.load_state_dict(torch.load(price_path, map_location=self.device))
            self.mode_model = ModeClassifier(input_size=3).to(self.device)
            self.mode_model.load_state_dict(torch.load(mode_path, map_location=self.device))
            self.price_model.eval(); self.mode_model.eval()
            logger.info('Loaded price & mode models and scalers from disk')
            return True
        except Exception as e:
            logger.exception('Failed to load models: %s', e)
            return False

    def create_sequences(self, data: np.ndarray, labels: Optional[np.ndarray] = None, is_mode: bool = False):
        X, y = [], []
        raw_close = getattr(self.data_fetcher, 'raw_close', None)
        close_idx = 3
        if hasattr(self.data_fetcher, 'feature_names') and 'close' in self.data_fetcher.feature_names:
            close_idx = self.data_fetcher.feature_names.index('close')
        look = max(1, self.config.LOOKBACK)
        for i in range(len(data) - look):
            X.append(data[i:i + look])
            if is_mode:
                idx = i + look - 1
                y.append(labels[idx] if (labels is not None and idx < len(labels)) else 0.0)
            else:
                if raw_close is not None and len(raw_close) >= i + look + 1:
                    next_close = raw_close[i + look]
                    last_close = raw_close[i + look - 1]
                    denom = last_close if abs(last_close) > 1e-12 else 1e-12
                    price_change = (next_close - last_close) / denom
                else:
                    next_close = data[i + look][close_idx]
                    last_close = data[i + look - 1][close_idx]
                    denom = last_close if abs(last_close) > 1e-8 else 1e-8
                    price_change = (next_close - last_close) / denom
                y.append(price_change)
        if len(X) == 0:
            return np.zeros((0, look, data.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def train(self, scaled_data: np.ndarray, mode_data: np.ndarray, mode_labels: np.ndarray):
        n_windows = 4
        if len(scaled_data) < self.config.LOOKBACK + 10:
            logger.warning('Not enough data to train price model properly; aborting training')
            return
        window_size = max(self.config.LOOKBACK + 5, len(scaled_data) // n_windows)
        val_losses = []
        for w in range(n_windows):
            end = min(len(scaled_data), (w + 1) * window_size)
            train_data = scaled_data[:end]
            test_data = scaled_data[end: min(len(scaled_data), end + window_size)]
            if len(train_data) <= self.config.LOOKBACK or len(test_data) <= self.config.LOOKBACK:
                continue
            X_train, y_train = self.create_sequences(train_data)
            X_test, y_test = self.create_sequences(test_data)
            if len(X_train) == 0 or len(X_test) == 0:
                continue
            tmp_model = LSTMModel(input_size=X_train.shape[2]).to(self.device)
            optimizer = optim.AdamW(tmp_model.parameters(), lr=1e-4, weight_decay=1e-5)
            crit = nn.SmoothL1Loss()

            train_loader = DataLoader(PriceDataset(X_train, y_train), batch_size=64, shuffle=True)
            val_loader = DataLoader(PriceDataset(X_test, y_test), batch_size=64, shuffle=False)
            best_val = float('inf')
            patience = 0
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3)
            for epoch in range(25):
                tmp_model.train()
                for bx, by in train_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    optimizer.zero_grad()
                    out = tmp_model(bx)
                    by = by.view(-1)
                    loss = crit(out, by)
                    loss.backward()
                    optimizer.step()
                tmp_model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx, by = bx.to(self.device), by.to(self.device)
                        preds = tmp_model(bx)
                        val_loss += crit(preds, by).item()
                val_loss = val_loss / max(1, len(val_loader))
                scheduler.step(val_loss)
                if val_loss < best_val - 1e-9:
                    best_val = val_loss; patience = 0
                else:
                    patience += 1
                if patience >= 5:
                    break
            val_losses.append(best_val)
            logger.info(f"Window {w} best val loss: {best_val:.6f}")
        logger.info(f"Walk-forward Price avg val loss: {np.mean(val_losses) if val_losses else float('nan')}")

        # Final train on all data
        X_all, y_all = self.create_sequences(scaled_data)
        if len(X_all) == 0:
            logger.error('Not enough sequences to train final model')
            return
        self.price_model = LSTMModel(input_size=X_all.shape[2]).to(self.device)
        self.price_optimizer = optim.AdamW(self.price_model.parameters(), lr=1e-4, weight_decay=1e-5)
        final_loader = DataLoader(PriceDataset(X_all, y_all), batch_size=64, shuffle=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.price_optimizer, mode='min', factor=0.7, patience=3)
        crit = nn.SmoothL1Loss()

        best_val = float('inf'); patience = 0
        for epoch in range(30):
            self.price_model.train()
            for bx, by in final_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                self.price_optimizer.zero_grad()
                out = self.price_model(bx)
                by = by.view(-1)
                loss = crit(out, by)
                loss.backward()
                self.price_optimizer.step()
            # no separate val here; break if needed via scheduler placeholder
        logger.info('Final price model trained on full dataset')

        # Mode model training
        if len(mode_data) > self.config.LOOKBACK + 5:
            n_windows_mode = 4
            window_size_mode = max(self.config.LOOKBACK + 5, len(mode_data) // n_windows_mode)
            mode_val_losses = []
            for w in range(n_windows_mode):
                end = min(len(mode_data), (w + 1) * window_size_mode)
                train_m = mode_data[:end]
                test_m = mode_data[end: min(len(mode_data), end + window_size_mode)]
                train_lab = mode_labels[:end]
                test_lab = mode_labels[end: min(len(mode_labels), end + window_size_mode)]
                if len(train_m) <= self.config.LOOKBACK or len(test_m) <= self.config.LOOKBACK:
                    continue
                X_m_train, y_m_train = self.create_sequences(train_m, train_lab, is_mode=True)
                X_m_test, y_m_test = self.create_sequences(test_m, test_lab, is_mode=True)
                if len(X_m_train) == 0 or len(X_m_test) == 0:
                    continue
                pos = float(np.sum(y_m_train)); neg = float(len(y_m_train) - pos)
                pos_weight_val = (neg / pos) if pos > 0 else 1.0
                pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=self.device)
                tmp_mode = ModeClassifier(input_size=X_m_train.shape[2]).to(self.device)
                opt = optim.Adam(tmp_mode.parameters(), lr=1e-3)
                crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                tr_loader = DataLoader(ModeDataset(X_m_train, y_m_train), batch_size=64, shuffle=True)
                val_loader = DataLoader(ModeDataset(X_m_test, y_m_test), batch_size=64, shuffle=False)
                best_val = float('inf'); patience = 0
                for epoch in range(20):
                    tmp_mode.train()
                    for bx, by in tr_loader:
                        bx, by = bx.to(self.device), by.to(self.device)
                        opt.zero_grad()
                        out = tmp_mode(bx)
                        by = by.view(-1)
                        loss = crit(out, by)
                        loss.backward()
                        opt.step()
                    tmp_mode.eval()
                    val_loss = 0.0; correct = 0; total = 0
                    with torch.no_grad():
                        for bx, by in val_loader:
                            bx, by = bx.to(self.device), by.to(self.device)
                            logits = tmp_mode(bx)
                            val_loss += crit(logits, by).item()
                            probs = torch.sigmoid(logits)
                            preds = (probs > 0.5).float()
                            correct += (preds == by).sum().item()
                            total += by.numel()
                    val_loss = val_loss / max(1, len(val_loader))
                    val_acc = correct / total if total > 0 else 0.0
                    if val_loss < best_val - 1e-9:
                        best_val = val_loss; patience = 0
                    else:
                        patience += 1
                    if patience >= 4:
                        break
                mode_val_losses.append(best_val)
                logger.info(f"Mode window {w} val_loss={best_val:.6f} val_acc={val_acc:.3f}")
            logger.info(f"Mode walk-forward avg val loss: {np.mean(mode_val_losses) if mode_val_losses else float('nan'):.6f}")

            X_mode_all, y_mode_all = self.create_sequences(mode_data, mode_labels, is_mode=True)
            if len(X_mode_all) > 0:
                pos = float(np.sum(y_mode_all)); neg = float(len(y_mode_all) - pos)
                pos_weight_val = (neg / pos) if pos > 0 else 1.0
                pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=self.device)
                self.mode_model = ModeClassifier(input_size=X_mode_all.shape[2]).to(self.device)
                self.mode_optimizer = optim.Adam(self.mode_model.parameters(), lr=1e-3)
                crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                mode_loader = DataLoader(ModeDataset(X_mode_all, y_mode_all), batch_size=64, shuffle=True)
                for epoch in range(30):
                    self.mode_model.train()
                    for bx, by in mode_loader:
                        bx, by = bx.to(self.device), by.to(self.device)
                        self.mode_optimizer.zero_grad()
                        out = self.mode_model(bx)
                        by = by.view(-1)
                        loss = crit(out, by)
                        loss.backward()
                        self.mode_optimizer.step()
                logger.info('Final mode model trained')
            else:
                logger.warning('Not enough mode data to train final mode model')
        else:
            logger.warning('Not enough mode data to train mode model')

        self.save_model()

    def predict_price(self, scaled_data: np.ndarray) -> Tuple[float, Tuple[float, float], float]:
        if self.price_model is None:
            raise RuntimeError('Price model not trained')
        #
        seq = scaled_data[-self.config.LOOKBACK:]
        if seq.shape[0] < self.config.LOOKBACK:
            raise ValueError('Not enough data for prediction sequence')
        preds = []
        x = torch.tensor(seq.reshape(1, seq.shape[0], seq.shape[1]), dtype=torch.float32).to(self.device)
        self.price_model.eval()
        with torch.no_grad():
            for _ in range(10):
                # MC-dropout style sampling: briefly enable train mode if model has dropout
                self.price_model.train()
                preds.append(self.price_model(x).cpu().numpy().item())
        mean_change = float(np.mean(preds))
        std_change = float(np.std(preds))
        last_close = float(self.data_fetcher.last_df['close'].iloc[-1])
        mean_price = last_close * (1.0 + mean_change)
        range_low = mean_price - std_change * abs(last_close)
        range_high = mean_price + std_change * abs(last_close)
        # return mean, (low,high), and std_change for gating downstream
        return mean_price, (range_low, range_high), std_change

    def predict_mode(self, mode_data: np.ndarray) -> str:
        if self.mode_model is None:
            return 'neutral'
        seq = mode_data[-self.config.LOOKBACK:]
        if seq.shape[0] < self.config.LOOKBACK:
            return 'neutral'
        with torch.no_grad():
            x = torch.tensor(seq.reshape(1, seq.shape[0], seq.shape[1]), dtype=torch.float32).to(self.device)
            logits = self.mode_model(x).cpu().numpy()
            logit = float(np.asarray(logits).ravel()[0])
            prob = 1.0 / (1.0 + np.exp(-logit))

            return 'aggressive' if prob > 0.5 else 'conservative'

    def predict(self, scaled_data: np.ndarray, mode_data: np.ndarray):
        if scaled_data is None or len(scaled_data) < self.config.LOOKBACK:
            return 'HOLD', 'neutral', float(self.data_fetcher.last_df['close'].iloc[-1])
        mean_price, _, std_change = self.predict_price(scaled_data)
        mode = self.predict_mode(mode_data)
        last_close = float(self.data_fetcher.last_df['close'].iloc[-1])
        atr = float(self.data_fetcher.last_df['ATR'].iloc[-1]) if 'ATR' in self.data_fetcher.last_df.columns else 0.0
        sens = getattr(self.data_fetcher, 'sensitivity', 1.0)

        # base threshold (small, scalping-oriented)
        threshold = max(atr * 0.05, abs(mean_price - last_close) * 0.05) * sens

        # uncertainty absolute in price units
        uncertainty_abs = std_change * abs(last_close)

        # require predicted move to exceed both threshold and a multiple of model uncertainty
        min_uncertainty_mult = 1.0  # require move > 1×std
        gating_value = max(threshold, uncertainty_abs * min_uncertainty_mult)

        price_diff = mean_price - last_close
        logger.info(f"Pred mean {mean_price:.5f} last {last_close:.5f} diff {price_diff:.6f} threshold {threshold:.6f} uncert_abs {uncertainty_abs:.6f} mode {mode}")

        # if prediction is not sufficiently larger than uncertainty, HOLD
        if abs(price_diff) < gating_value:
            logger.info(f"HOLD by gating: |price_diff| {abs(price_diff):.6f} < gating_value {gating_value:.6f}")
            return 'HOLD', mode, float(mean_price)

        # otherwise decide direction
        if price_diff > 0:
            return 'BUY', mode, float(mean_price)
        else:
            return 'SELL', mode, float(mean_price)

# ---------- Executor ----------
class Executor:
    def __init__(self, config: Config, data_fetcher: DataFetcher):
        self.config = config
        self.data_fetcher = data_fetcher
        self.pending_orders = {}
        self.processed_deals = set()
        self.recent_pnls = []
        self.sensitivity = 1.0

    def has_open_position(self) -> bool:
        if mt5 is None:
            return False
        try:
            positions = mt5.positions_get(symbol=self.config.SYMBOL)
            return bool(positions and len(positions) > 0)
        except Exception:
            return False

    def place_order(self, action: str, pred_price: float, mode: str, win_rate: float = 0.55, rr: float = 2.0):
        if action not in ('BUY', 'SELL'):
            logger.info(f"Action {action} -> no order placed.")
            return None
        try:
            if self.has_open_position():
                logger.info('Already have open position; skipping new order')
                return None
            if mt5 is None:
                logger.info('MT5 not available; dry-run only. Not sending order.')
                logger.info(f'DRYRUN: {action} at predicted {pred_price:.5f} mode {mode}')
                return None

            # load market & account info
            symbol_info = mt5.symbol_info(self.config.SYMBOL)
            if symbol_info is None:
                logger.warning('Symbol info missing from MT5'); return None

            # current market tick
            tick = mt5.symbol_info_tick(self.config.SYMBOL)
            if tick is None:
                logger.warning('Failed to fetch symbol tick'); return None

            # last dataframe and ATR (for spread-rel checks)
            df = self.data_fetcher.last_df
            if df is None or df.empty:
                logger.warning('No market data for SL calc'); return None
            atr = compute_atr(df['high'], df['low'], df['close'], self.config.ATR_PERIOD).iloc[-1]
            if atr is None or np.isnan(atr) or atr <= 0:
                logger.warning('Invalid ATR; skipping order'); return None

            # spread check (skip if too wide relative to ATR)
            spread = abs(float(tick.ask) - float(tick.bid))
            if spread > max(atr * 0.2, 0.1):   # 0.5 USD floor for XAUUSD; tune per broker
                logger.info(f"Spread too wide ({spread:.5f}), skipping order")
                return None

            price = float(tick.ask) if action == 'BUY' else float(tick.bid)

            if action == 'BUY':
                tp = float(pred_price - max((pred_price - price) * 0.05, atr * 0.1))
                sl = float(price - atr)
                order_type = mt5.ORDER_TYPE_BUY
            else:
                tp = float(pred_price + max((price - pred_price) * 0.05, atr * 0.1))
                sl = float(price + atr)
                order_type = mt5.ORDER_TYPE_SELL
            account_info = mt5.account_info()
            if account_info is None:
                logger.warning('Account info missing'); return None
            account_balance = float(account_info.balance)
            kelly_fraction = (win_rate * (rr + 1) - 1) / rr
            kelly_fraction = max(0.01, min(kelly_fraction, 0.1))
            risk_amount = account_balance * kelly_fraction
            point = getattr(symbol_info, 'point', 0.01)
            sl_pips = max(1e-8, abs(price - sl) / point)
            denom = sl_pips * point * 100000.0
            if abs(denom) < 1e-8:
                volume = self.config.LOT_SIZE
            else:
                volume = risk_amount / denom
                volume = max(self.config.LOT_SIZE, min(volume, self.config.MAX_LOT))
                vol_step = getattr(symbol_info, 'volume_step', 0.01)
                if vol_step > 0:
                    volume = float(round(volume / vol_step) * vol_step)
            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': self.config.SYMBOL,
                'volume': float(volume),
                'type': order_type,
                'price': price,
                'sl': sl,
                'tp': tp,
                'deviation': 20,
                'magic': 234000,
                'comment': 'AI Bot',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result is None:
                logger.warning('MT5 order_send returned None'); return None
            retcode = getattr(result, 'retcode', None)
            if retcode != mt5.TRADE_RETCODE_DONE:
                logger.warning(f'Order failed with retcode {retcode}'); return None
            logger.info(f'Order placed: {action} vol={volume:.4f} sl={sl:.5f} tp={tp:.5f}')
            order_ticket = getattr(result, 'order', None) or getattr(result, 'deal', None) or None
            if order_ticket is not None:
                self.pending_orders[str(order_ticket)] = {
                    'open_price': price, 'volume': volume, 'action': action, 'tp': tp, 'sl': sl, 'timestamp': datetime.now(timezone.utc)
                }
            return result
        except Exception as e:
            logger.exception('Exception in place_order: %s', e)
            return None

    def check_closed_orders(self):
        if mt5 is None:
            return
        try:
            to_utc = datetime.now(timezone.utc)
            from_utc = to_utc - timedelta(days=7)
            deals = mt5.history_deals_get(from_utc, to_utc)
            if not deals:
                return
            for d in deals:
                deal_id = getattr(d, 'deal', None) or getattr(d, 'ticket', None)
                if deal_id is None:
                    continue
                if deal_id in self.processed_deals:
                    continue
                comment = getattr(d, 'comment', '') or ''
                if 'AI Bot' in comment or getattr(d, 'magic', None) == 234000:
                    pnl = getattr(d, 'profit', None)
                    if pnl is not None:
                        try:
                            pnl = float(pnl)
                            self.recent_pnls.append(pnl)
                            if len(self.recent_pnls) > 10:
                                self.recent_pnls.pop(0)
                            avg_pnl = float(np.mean(self.recent_pnls))
                            sens = getattr(self.data_fetcher, 'sensitivity', 1.0)
                            sens = sens * (0.90 if avg_pnl > 0 else 1.10)
                            sens = float(np.clip(sens, 0.5, 2.0))
                            self.data_fetcher.sensitivity = sens
                            logger.info(f"Updated sensitivity to {sens:.3f} from avg pnl {avg_pnl:.3f}")
                        except Exception:
                            pass
                    self.processed_deals.add(deal_id)
        except Exception as e:
            logger.exception('Error in check_closed_orders: %s', e)

# ---------- Synthetic data generator ----------
def generate_synthetic_data(bars: int = 2000, start_price: float = 2000.0) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    dt = pd.date_range(end=pd.Timestamp.utcnow(), periods=bars, freq='5min')
    price = start_price
    rows = []
    for t in dt:
        change = rng.normal(loc=0.00002, scale=0.0006)
        new_price = price * (1 + change)
        high = max(price, new_price) * (1 + abs(rng.normal(0, 0.0001)))
        low = min(price, new_price) * (1 - abs(rng.normal(0, 0.0001)))
        open_p = price
        close_p = new_price
        tick_vol = int(abs(rng.normal(100, 30)))
        rows.append({'time': t, 'open': open_p, 'high': high, 'low': low, 'close': close_p, 'tick_volume': tick_vol})
        price = new_price
    df = pd.DataFrame(rows)
    return df

# ---------- Main ----------
def main(args):
    cfg = Config()
    dfetch = DataFetcher(cfg)
    trainer = ModelTrainer(cfg, dfetch)
    execer = Executor(cfg, dfetch)

    if args.test:
        logger.info('Running in self-test mode with synthetic data')
        df = generate_synthetic_data(bars=4000, start_price=2000.0)
        scaled = dfetch.preprocess_data(df, fit_scaler=True)
        mode_data, mode_labels = dfetch.prepare_mode_data(df, fit_scaler=True)
        trainer.train(scaled, mode_data, mode_labels)
        if trainer.price_model is not None:
            try:
                action, mode, pred_price = trainer.predict(scaled, mode_data)
                logger.info(f'SMOKE TEST PREDICT -> Action: {action}, Mode: {mode}, Pred: {pred_price:.5f}')
            except Exception as e:
                logger.exception('Prediction failed in test: %s', e)
        logger.info('Self-test complete.')
        return

    if mt5 is None:
        logger.warning('MT5 library not present. Use --test for offline run')
        return

    if not dfetch.connect_mt5():
        logger.warning('Could not initialize MT5. Abort.')
        return

    try:
        df_hist = dfetch.fetch_rates(cfg.HISTORY_BARS)
    except Exception as e:
        logger.exception('Failed to fetch MT5 rates: %s', e)
        return

    loaded = trainer.try_load_models()
    if not loaded:
        try:
            scaled = dfetch.preprocess_data(df_hist, fit_scaler=True)
            mode_data, mode_labels = dfetch.prepare_mode_data(df_hist, fit_scaler=True)
        except Exception as e:
            logger.exception('Preprocessing failed: %s', e)
            return
        trainer.train(scaled, mode_data, mode_labels)
        loaded = True
    else:
        try:
            scaled = dfetch.preprocess_data(df_hist, fit_scaler=False)
            mode_data, mode_labels = dfetch.prepare_mode_data(df_hist, fit_scaler=False)
        except Exception as e:
            logger.exception('Preprocessing with loaded scalers failed: %s', e)
            return

    logger.info('Entering live trading loop (dry-run unless --live passed)')
    try:
        while True:
            try:
                df_now = dfetch.fetch_multi_timeframe(bars_m5=500)
                # use fit_scaler=False because we either loaded scalers or just trained and saved above
                scaled_now = dfetch.preprocess_data(df_now, fit_scaler=not loaded)
                mode_now, _ = dfetch.prepare_mode_data(df_now, fit_scaler=not loaded)
                action, mode, pred_price = trainer.predict(scaled_now, mode_now)
                logger.info(f'Live decision: {action} (mode={mode}) pred={pred_price:.5f}')
                if args.live and action in ('BUY', 'SELL'):
                    execer.place_order(action, pred_price, mode)
                else:
                    logger.info('Not placing order (dry-run or HOLD).')
                execer.check_closed_orders()
            except Exception as e:
                logger.exception('Error during live loop iteration: %s', e)
            time.sleep(getattr(cfg, 'LIVE_LOOP_SLEEP', 60))
    except KeyboardInterrupt:
        logger.info('Interrupted by user; exiting.')
    except Exception as e:
        logger.exception('Unhandled in live loop: %s', e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run self-test on synthetic data (no MT5)')
    parser.add_argument('--live', action='store_true', help='Enable live trading via MT5 (use with caution)')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        logger.exception('Unhandled exception in main: %s', e)
        sys.exit(1)
