"""
=============================================================
  SCREENER AUTOMATISÉ SERVEUR — SANS BOT / VIA REQUESTS
=============================================================
Scan quotidien ultra-léger sans UI.
Évalue les opportunités sur la clôture du jour.
Envoie une alerte Telegram directe via un appel HTTP requests.
=============================================================
"""

import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# ──────────────────────────────────────────────────────────────
# CONFIGURATION TELEGRAM & ENVIRONNEMENT SERVEUR
# ──────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = "8754756145:AAEHQ94Y52WcRl7nL3Bzr8naEkM5VRVdKdg"
TELEGRAM_CHAT_ID = "7120971309"

with open("listes.json", "r") as f:
    watchlist_data = json.load(f)


# Nom de la liste du fichier JSON à utiliser pour le scan quotidien
WATCHLIST = watchlist_data[0]["liste"]

# ══════════════════════════════════════════════════════════════
# 1. DATACLASS STRATEGYCONFIG
# ══════════════════════════════════════════════════════════════
@dataclass
class StrategyConfig:
    name: str = "Default"
    stop_loss_pct: float = 0.15
    take_profit_pct: float = 0.10
    type_vente: str = "SUIVEUR_SECURISE"
    secure_trigger_pct: float = 0.10
    minimum_score: int = 60

    tests: Dict[str, int] = field(default_factory=lambda: {
        "ABOVE_MA200":      20,
        "RSI_MOMENTUM":     20,
        "RSI_OVERSOLD":     15,
        "MACD_SIGNAL":      20,
        "MA20>MA50>MA200":  35,
        "BOLLINGER_BREAK":  10,
        "VOLUME_SURGE":     10,
        "ATR_FAVORABLE":    10,
        "TREND_SLOPE":      10,
        "CLOSE_TO_MA200":    5,
    })

    rsi_momentum_min: float = 50.0
    rsi_oversold_max: float = 35.0
    volume_ratio_min: float = 1.5
    atr_pct_max: float = 0.03
    trend_slope_days: int = 10
    close_to_ma200_pct: float = 0.05

    def total_possible_score(self) -> int:
        return sum(self.tests.values())

    def normalized_minimum(self) -> float:
        total = self.total_possible_score()
        return (self.minimum_score / 100.0) * total if total > 0 else 0


# Tes stratégies cibles pour le trading réel
STRATEGY_CATALOG: List[StrategyConfig] = [
    StrategyConfig(
        name="🚀 Breakout Volume",
        stop_loss_pct=0.15, take_profit_pct=0.20, type_vente="SUIVEUR",
        minimum_score=75, volume_ratio_min=2.0,
        tests={"ABOVE_MA200": 15, "BOLLINGER_BREAK": 25, "VOLUME_SURGE": 30, "MACD_SIGNAL": 15, "TREND_SLOPE": 15}
    )
]


# ══════════════════════════════════════════════════════════════
# 2. CALCUL DES INDICATEURS AVEC LISSAGE DE WILDER
# ══════════════════════════════════════════════════════════════
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.copy()

    df['MA200'] = df['Close'].rolling(200).mean()
    df['MA50']  = df['Close'].rolling(50).mean()
    df['MA20']  = df['Close'].rolling(20).mean()

    # RSI de Wilder
    delta = df['Close'].diff()
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)
    gain_smooth = gains.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss_smooth = losses.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + gain_smooth / loss_smooth.replace(0, np.nan)))

    df['MACD']   = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    bb_mid = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_Upper'] = bb_mid + 2 * bb_std
    df['BB_Lower'] = bb_mid - 2 * bb_std

    # ATR de Wilder
    high_low = df['High'] - df['Low']
    high_cp  = (df['High'] - df['Close'].shift(1)).abs()
    low_cp   = (df['Low']  - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()

    return df


# ══════════════════════════════════════════════════════════════
# 3. NOTATION / SCORING TEMPS RÉEL
# ══════════════════════════════════════════════════════════════
def score_row(cfg: StrategyConfig, row, prev_row, df_hist=None) -> Tuple[int, List[str]]:
    score = 0
    tests_valides = []

    def _get(series_name, default=None):
        try:
            val = float(row[series_name])
            return val if not np.isnan(val) else default
        except Exception:
            return default

    close   = _get('Close')
    ma200   = _get('MA200')
    ma50    = _get('MA50')
    ma20    = _get('MA20')
    rsi     = _get('RSI')
    macd    = _get('MACD')
    signal  = _get('Signal')
    bb_up   = _get('BB_Upper')
    atr     = _get('ATR')
    vol     = _get('Volume')
    vol_ma  = _get('Vol_MA20')
    prev_rsi = float(prev_row['RSI']) if prev_row is not None and not np.isnan(float(prev_row['RSI'])) else None

    above_ma200 = (close is not None and ma200 is not None and close > ma200)

    for test_name, weight in cfg.tests.items():
        if weight == 0:
            continue

        passed = False
        if test_name == "ABOVE_MA200":
            passed = above_ma200
        elif test_name == "RSI_MOMENTUM":
            passed = (rsi is not None and prev_rsi is not None and rsi > cfg.rsi_momentum_min and rsi > prev_rsi)
        elif test_name == "RSI_OVERSOLD":
            passed = (rsi is not None and rsi < cfg.rsi_oversold_max)
        elif test_name == "MACD_SIGNAL":
            passed = (macd is not None and signal is not None and macd > signal)
        elif test_name == "MA20>MA50>MA200":
            passed = (ma20 is not None and ma50 is not None and ma200 is not None and ma20 > ma50 > ma200)
        elif test_name == "BOLLINGER_BREAK":
            passed = (close is not None and bb_up is not None and close > bb_up)
        elif test_name == "VOLUME_SURGE":
            passed = (vol is not None and vol_ma is not None and vol_ma > 0 and (vol / vol_ma) >= cfg.volume_ratio_min)
        elif test_name == "ATR_FAVORABLE":
            passed = (atr is not None and close is not None and close > 0 and (atr / close) <= cfg.atr_pct_max)
        elif test_name == "TREND_SLOPE" and df_hist is not None and ma50 is not None:
            try:
                idx_now = df_hist.index.get_loc(row.name)
                if idx_now >= cfg.trend_slope_days:
                    ma50_past = float(df_hist['MA50'].iloc[idx_now - cfg.trend_slope_days])
                    passed = (not np.isnan(ma50_past) and ma50 > ma50_past)
            except Exception: pass
        elif test_name == "CLOSE_TO_MA200":
            passed = (close is not None and ma200 is not None and ma200 > 0 and (abs(close - ma200) / ma200) <= cfg.close_to_ma200_pct)

        if passed:
            score += weight
            tests_valides.append(f"{test_name} (+{weight}pts)")

    return score, tests_valides


# ══════════════════════════════════════════════════════════════
# 4. EXPÉDITION TELEGRAM (REQUÊTE DIRECTE VIA API HTTP)
# ══════════════════════════════════════════════════════════════
def send_telegram_alert(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&text={message}"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"❌ Échec Telegram ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"❌ Erreur réseau lors de l'appel Telegram : {e}")


# ══════════════════════════════════════════════════════════════
# 5. POINT D'ENTRÉE DU SCREENER DAILY
# ══════════════════════════════════════════════════════════════
def run_daily_screener():
   # Extraction de la watchlist
   
    print(f"🔍 Scan quotidien actif sur {len(WATCHLIST)} actions...")
    opportunites = []

    for ticker in WATCHLIST:
        try:
            # Récupération des deux dernières années (ajuste au besoin)
            df = yf.download(ticker, period="2y", progress=False)
            if df.empty or len(df) < 201:
                continue

            df = compute_indicators(df)
            row = df.iloc[-1]
            prev_row = df.iloc[-2]

            for cfg in STRATEGY_CATALOG:
                score, tests_actifs = score_row(cfg, row, prev_row, df)
                score_normalise = (score / cfg.total_possible_score()) * 100

                # Vérification du déclenchement
                if score_normalise >= cfg.minimum_score:
                    opportunites.append({
                        "ticker": ticker,
                        "strategie": cfg.name,
                        "score": round(score_normalise, 1),
                        "prix": round(float(row['Close']), 2),
                        "rsi": round(float(row['RSI']), 1) if not np.isnan(row['RSI']) else "N/A",
                        "tests": tests_actifs
                    })
        except Exception as e:
            print(f"⚠️ Erreur lors de l'analyse de {ticker}: {e}")
            continue

    # Envoi du récapitulatif groupé s'il y a des signaux valides
    if opportunites:
        for opp in opportunites:
            msg = f"🔔 *SCREENER ALERT — {datetime.now().strftime('%d/%m/%Y')}* 🔔\n\n"
            msg += f"🎯 *{opp['ticker']}* détecté !\n"
            msg += f" ├ 📊 Stratégie : {opp['strategie']}\n"
            msg += f" ├ 🏆 Score : *{opp['score']}%* (Min {opp['score']:.0f}%)\n"
            msg += f" ├ 💵 Prix : {opp['prix']} €\n"
            msg += f" ├ 📈 RSI : {opp['rsi']}\n"
            msg += f" └ 🛠️ Filtres validés : _{', '.join(opp['tests'])}_\n\n"
            msg += f"────────────────────\n\n"
            send_telegram_alert(msg)
        print("📨 Opportunités validées. Envoi de la notification...")
        
    else:
        print("😴 Fin du scan. Aucun signal identifié aujourd'hui.")


if __name__ == "__main__":
    import urllib.request
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')]
    urllib.request.install_opener(opener)
    run_daily_screener()
