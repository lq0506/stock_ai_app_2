import os
import sys
import warnings
import streamlit as st
from datetime import datetime
import pandas as pd
import baostock as bs
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import threading
from queue import Queue

# ==================== 全局配置 ====================
SEQ_WINDOW = 15
TRAIN_RATIO = 0.8
EPOCHS = 300
PATIENCE = 20
today = datetime.today().strftime("%Y-%m-%d")
MODEL_FILE = "temp_best_model.pth"

# EXPMA 参数
EXP_SHORT1 = 5
EXP_SHORT2 = 10
EXP_BAND1 = 12
EXP_BAND2 = 26
EXP_TREND1 = 20
EXP_TREND2 = 60

# 点位计算回溯周期
PRICE_LOOKBACK = 20

# 运行锁标记
is_running = False
log_queue = Queue(maxsize=500)

# ========== 子线程日志 ==========
def add_log_bg(msg):
    log_queue.put(msg)

# ===================== 工具函数【原样不动】 =====================
def get_code(code):
    if len(code) != 6 or not code.isdigit():
        return ""
    if code.startswith(("600", "601", "603", "605")):
        return f"sh.{code}"
    else:
        return f"sz.{code}"

def calc_expma(series, n):
    alpha = 2 / (n + 1)
    expma = series.ewm(alpha=alpha, adjust=False).mean()
    return expma

def calc_kdj(df, n=9, m1=3, m2=3):
    low_min = df["low"].rolling(n).min()
    high_max = df["high"].rolling(n).max()
    rsv = (df["close"] - low_min) / (high_max - low_min + 1e-6) * 100
    df["k"] = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    df["d"] = df["k"].ewm(alpha=1 / m2, adjust=False).mean()
    df["j"] = 3 * df["k"] - 2 * df["d"]
    return df

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    return 100 - 100 / (1 + rs)

def calc_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["diff"] = ema_fast - ema_slow
    df["dea"] = df["diff"].ewm(span=signal, adjust=False).mean()
    df["macd"] = (df["diff"] - df["dea"]) * 2
    return df

def create_sequences(data, label_up, label_in, window):
    X, y_up, y_in = [], [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y_up.append(label_up[i + window - 1])
        y_in.append(label_in[i + window - 1])
    return np.array(X), np.array(y_up), np.array(y_in)

# ===================== 指标打分【原样不动】 =====================
def calc_indicator_score(row):
    score = 0.0
    score += 10 if row["ma5_above_ma20"] else -10
    score += 8 if row["price_above_ma20"] else -8
    score += 4 if row["price_above_ma5"] else -4
    if row["kdj_over_sell"]:
        score += 15
    elif row["kdj_over_buy"]:
        score -= 10
    if row["macd_gold"] and row["rsi14"] < 60:
        score += 18
    elif row["macd_dead"]:
        score -= 25
    if row["rsi_over_sell"]:
        score += 15
    elif row["rsi_over_buy"]:
        score -= 20
    if row["exp_bull"]:
        if row["rsi_over_buy"] or row["kdj_over_buy"]:
            score += 22
        else:
            score += 25
    elif row["exp_dead"]:
        score -= 30
    if row["close"] > row["exp60"] and not (row["rsi_over_buy"] or row["kdj_over_buy"]):
        score += 12
    elif row["close"] < row["exp20"] and not(row["rsi_over_sell"] or row["kdj_over_sell"]):
        score -= 15
    if row["model_prob_in"] > 0.6:
        score += 20
    elif row["model_prob_in"] < 0.3:
        score -= 12
    return np.clip(score / 120, -1.0, 1.0)

def hard_constraint(ret):
    if ret < -0.03:
        return 0.25
    elif ret < -0.01:
        return 0.4
    elif ret > 0.03:
        return 0.92
    elif ret > 0.01:
        return 0.8
    return 0.7

def final_prob(row):
    model_base = row["model_prob_up"]
    tech_score = 0.5 + row["indicator_score"] * 0.5
    raw_prob = model_base * 0.15 + tech_score * 0.85
    k_over = row["kdj_over_buy"]
    r_over = row["rsi_over_buy"]
    bull_exp = row["exp_bull"]
    if bull_exp:
        if k_over and r_over:
            raw_prob = max(raw_prob - 0.25, 0.05)
        elif k_over:
            raw_prob = max(raw_prob - 0.02, 0.05)
    if row["rsi_over_sell"] and row["kdj_over_sell"]:
        raw_prob = min(raw_prob + 0.28, 0.95)
    elif row["rsi_over_sell"] or row["kdj_over_sell"]:
        raw_prob = min(raw_prob + 0.15, 0.95)
    if row["ret"] > 0.07:
        raw_prob = min(raw_prob + 0.06, 0.95)
    max_p = hard_constraint(row["ret"])
    prob = min(raw_prob, max_p)
    return float(np.clip(prob, 0.05, 0.95))

def get_status(p, ret, gap_up, consec_up):
    if ret < -0.02:
        return "💸资金流出"
    if consec_up >= 2 and gap_up > 2.5 or gap_up>3 or ret>0.07:
        return "🔥强势流入"
    if p > 0.6:
        return "🔥强势流入"
    if p > 0.35:
        return "✅温和流入"
    if p < 0.2:
        return "💸资金流出"
    return "正常"

def get_kdj(row):
    return "🟥超买" if row["kdj_over_buy"] else "🟩超卖" if row["kdj_over_sell"] else "正常"
def get_rsi(row):
    return "🟥超买" if row["rsi_over_buy"] else "🟩超卖" if row["rsi_over_sell"] else "正常"
def get_macd(row):
    return "✅金叉" if row["macd_gold"] else "❌死叉" if row["macd_dead"] else "中性"
def get_ma(row):
    return "📈多头" if row["ma5_above_ma20"] else "📉空头"
def get_exp_status(row):
    if row["exp_gold"]:
        return "✅EXPMA金叉"
    elif row["exp_dead"]:
        return "❌EXPMA死叉"
    elif row["exp_bull"]:
        return "📊多头排列"
    else:
        return "➖震荡整理"

# ===================== 点位计算【原样不动】 =====================
def calc_price_levels(df, lookback=20):
    if df.empty or len(df) < lookback:
        return None
    df_slice = df.tail(lookback).copy()
    close = round(df_slice["close"].iloc[-1], 2)
    open_price = df_slice["open"].iloc[-1]
    exp20 = df_slice["exp20"].iloc[-1] if "exp20" in df_slice.columns else close
    exp60 = df_slice["exp60"].iloc[-1] if "exp60" in df_slice.columns else close
    boll_up = df_slice["boll_up"].iloc[-1] if "boll_up" in df_slice.columns else close * 1.08
    boll_down = df_slice["boll_down"].iloc[-1] if "boll_down" in df_slice.columns else close * 0.92
    recent_high = df_slice["high"].max()
    recent_low = df_slice["low"].min()
    def calc_atr(data, period=14):
        high = data["high"]
        low = data["low"]
        pre_close = data["close"].shift(1)
        tr1 = high - low
        tr2 = abs(high - pre_close)
        tr3 = abs(low - pre_close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr.iloc[-1] if not np.isnan(atr.iloc[-1]) else close * 0.03
    atr = calc_atr(df_slice, 14)
    day_chg = (close - open_price) / open_price if open_price != 0 else 0
    is_strong = (close > exp20) and day_chg > 0.02
    is_weak = (close < exp20) and day_chg < -0.015
    is_normal = not is_strong and not is_weak
    if is_strong:
        support_candidates = [exp20, close * 0.97]
    elif is_weak:
        support_candidates = [recent_low, boll_down, exp60]
    else:
        support_candidates = [exp20, boll_down, close * 0.96]
    support_raw = max([x for x in support_candidates if x < close], default=close * 0.95)
    support = max(support_raw, close * 0.92)
    support = round(support, 2)
    if is_strong:
        best_buy = close * 0.985
    elif is_weak:
        best_buy = support * 1.02
    else:
        best_buy = support + (close - support) * 0.35
    best_buy = min(best_buy, close * 0.99)
    best_buy = max(best_buy, support * 1.01)
    best_buy = round(best_buy, 2)
    pressure_short_raw = min(close * 1.04, boll_up, recent_high)
    pressure_short = max(pressure_short_raw, close * 1.02)
    pressure_short = round(pressure_short, 2)
    pressure_long_raw = max(boll_up, recent_high)
    pressure_long = max(pressure_long_raw, pressure_short)
    pressure_long = round(pressure_long, 2)
    sell_ref = max(close * 1.03, pressure_short * 0.97)
    sell_ref = min(sell_ref, pressure_short * 0.99)
    sell_ref = round(sell_ref, 2)
    if is_weak:
        stop_base = support - atr * 1.0
    else:
        stop_base = support - atr * 0.8
    max_stop = close * 0.94 if is_weak else close * 0.95
    stop_loss = max(stop_base, max_stop)
    stop_loss = min(stop_loss, support * 0.98)
    stop_loss = round(stop_loss, 2)
    if best_buy <= support:
        best_buy = round(support * 1.01, 2)
    if best_buy >= close:
        best_buy = round(close * 0.98, 2)
    if stop_loss >= support:
        stop_loss = round(support * 0.96, 2)
    if sell_ref <= close:
        sell_ref = round(close * 0.92, 2)
    if sell_ref >= pressure_short:
        sell_ref = round(pressure_short * 0.98, 2)
    if pressure_short >= pressure_long:
        pressure_long = round(pressure_short * 1.02, 2)
    if best_buy <= stop_loss:
        best_buy = round(stop_loss * 1.02, 2)
    return {
        "close": close,
        "best_buy": best_buy,
        "support": support,
        "pressure": pressure_short,
        "pressure_long": pressure_long,
        "stop_loss": stop_loss,
        "sell_ref": sell_ref
    }

# ===================== 模型定义【原样不动】 =====================
class CNNLSTM(nn.Module):
    def __init__(self, feat_dim, seq_len):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(feat_dim, 64, 3, padding=1), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(64, 32, 3, padding=1), nn.ReLU(), nn.Dropout(0.2)
        )
        self.lstm = nn.LSTM(32, 16, 2, batch_first=True, dropout=0.2)
        self.fc_up = nn.Linear(16, 1)
        self.fc_in = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.cnn(x).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        return self.sigmoid(self.fc_up(out)), self.sigmoid(self.fc_in(out))

# ===================== 核心运行（手机端优化排版） =====================
def run_analysis(stock_code, run_wfo):
    global is_running
    add_log_bg("# 🔍 正在深度量化分析股票数据\n> 模型运算耗时较长，请勿重复点击，等待测算完成\n")
    code = get_code(stock_code.strip())
    if not code:
        add_log_bg("❌ **错误：请输入正确6位数字股票代码**")
        is_running = False
        return
    try:
        bs.login()
        start_date = "2018-01-01"
        end_date = today
        rs = bs.query_history_k_data_plus(code=code,fields="date,open,high,low,close,volume,amount,pctChg",start_date=start_date, end_date=end_date, frequency="d")
        data = []
        while rs.error_code == '0' and rs.next():
            data.append(rs.get_row_data())
        df = pd.DataFrame(data, columns=rs.fields)
        bs.logout()
        if len(df) < 80:
            add_log_bg("⚠️ **警告：历史数据过少，无法完成指标测算**")
            is_running = False
            return
    except Exception as e:
        add_log_bg(f"❌ **数据获取异常：{str(e)}**")
        is_running = False
        return
    num_cols = ["open", "high", "low", "close", "volume", "amount", "pctChg"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    add_log_bg("✅ 数据获取完成")

    # 指标计算全流程不变
    df["prev_close"] = df["close"].shift(1)
    df["ret"] = df["close"].pct_change()
    df["gap_up"] = (df["open"] - df["prev_close"]) / df["prev_close"] * 100
    df["intra_strength"] = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-6)
    df["vol_ma5"] = df["volume"].rolling(5).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma5"]
    df["amount_ma5"] = df["amount"].rolling(5).mean()
    df["amount_ratio"] = df["amount"] / df["amount_ma5"]
    df["up_day"] = (df["ret"] > 0).astype(int)
    df["consec_up"] = df.groupby((df["up_day"] != df["up_day"].shift(1)).cumsum())["up_day"].cumsum()
    df["limit_up"] = (df["ret"] > 0.09).astype(int)
    df["strong_up"] = (df["ret"] > 0.02).astype(int)
    df["big_vol"] = (df["vol_ratio"] > 1.1).astype(int)
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma5_above_ma20"] = (df["ma5"] > df["ma20"]).astype(int)
    df["price_above_ma5"] = (df["close"] > df["ma5"]).astype(int)
    df["price_above_ma20"] = (df["close"] > df["ma20"]).astype(int)
    df["bias_ma5"] = (df["close"] - df["ma5"]) / (df["ma5"] + 1e-6)
    df["bias_ma20"] = (df["close"] - df["ma20"]) / (df["ma20"] + 1e-6)
    df["exp5"]  = calc_expma(df["close"], EXP_SHORT1)
    df["exp10"] = calc_expma(df["close"], EXP_SHORT2)
    df["exp12"] = calc_expma(df["close"], EXP_BAND1)
    df["exp26"] = calc_expma(df["close"], EXP_BAND2)
    df["exp20"] = calc_expma(df["close"], EXP_TREND1)
    df["exp60"] = calc_expma(df["close"], EXP_TREND2)
    df["exp_gold"] = ((df["exp12"] > df["exp26"]) & (df["exp12"].shift(1) <= df["exp26"].shift(1))).astype(int)
    df["exp_dead"] = ((df["exp12"] < df["exp26"]) & (df["exp12"].shift(1) >= df["exp26"].shift(1))).astype(int)
    df["exp_bull"] = ((df["exp5"] > df["exp10"]) & (df["exp10"] > df["exp20"]) & (df["exp20"] > df["exp60"])).astype(int)
    df = calc_kdj(df)
    df["kdj_over_buy"] = (df["j"] > 85).astype(int)
    df["kdj_over_sell"] = (df["j"] < 20).astype(int)
    df["rsi14"] = calc_rsi(df["close"])
    df["rsi_over_buy"] = (df["rsi14"] > 70).astype(int)
    df["rsi_over_sell"] = (df["rsi14"] < 30).astype(int)
    df = calc_macd(df)
    df["macd_gold"] = ((df["diff"] > df["dea"]) & (df["diff"].shift(1) <= df["dea"].shift(1))).astype(int)
    df["macd_dead"] = ((df["diff"] < df["dea"]) & (df["diff"].shift(1) >= df["dea"].shift(1))).astype(int)
    df["boll_mid"] = df["close"].rolling(20).mean()
    df["boll_std"] = df["close"].rolling(20).std()
    df["boll_up"] = df["boll_mid"] + 2 * df["boll_std"]
    df["boll_down"] = df["boll_mid"] - 2 * df["boll_std"]
    df["price_touch_up"] = (df["close"] >= df["boll_up"]).astype(int)
    df["price_touch_down"] = (df["close"] <= df["boll_down"]).astype(int)
    feats = [
        "gap_up", "ret", "intra_strength", "vol_ratio", "amount_ratio", "consec_up", "limit_up",
        "ma5_above_ma20", "price_above_ma5", "price_above_ma20", "bias_ma5", "bias_ma20",
        "k", "d", "j", "kdj_over_buy", "kdj_over_sell",
        "rsi14", "rsi_over_buy", "rsi_over_sell",
        "diff", "dea", "macd", "macd_gold", "macd_dead",
        "price_touch_up", "price_touch_down", "strong_up", "big_vol",
        "exp5","exp10","exp12","exp26","exp20","exp60","exp_gold","exp_dead","exp_bull"
    ]
    df = df.dropna(subset=feats).reset_index(drop=True)
    add_log_bg("✅ 全量指标计算完成")
    df["ret_tomorrow"] = df["ret"].shift(-1)
    df["vol_tomorrow_ratio"] = df["volume"].shift(-1) / df["vol_ma5"].shift(-1)
    df["label_up"] = ((df["ret_tomorrow"] > 0.015) & (df["vol_tomorrow_ratio"] > 1.05)).astype(int)
    df["label_in"] = ((df["consec_up"] >= 2) | (df["gap_up"] > 3) | ((df["ret"] > 0.01) & (df["vol_ratio"] > 1.05))).astype(int)
    scaler = StandardScaler()
    feat_data = scaler.fit_transform(df[feats])
    X_all, y_up_all, y_in_all = create_sequences(feat_data, df["label_up"].values, df["label_in"].values, SEQ_WINDOW)
    split_idx = int(len(X_all) * TRAIN_RATIO)
    X_train, X_val = X_all[:split_idx], X_all[split_idx:]
    y_up_train, y_up_val = y_up_all[:split_idx], y_up_all[split_idx:]
    y_in_train, y_in_val = y_in_all[:split_idx], y_in_all[split_idx:]
    X_train_t = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1)
    y_up_train_t = torch.tensor(y_up_train, dtype=torch.float32).unsqueeze(1)
    y_in_train_t = torch.tensor(y_in_train, dtype=torch.float32).unsqueeze(1)
    y_up_val_t = torch.tensor(y_up_val, dtype=torch.float32).unsqueeze(1)
    y_in_val_t = torch.tensor(y_in_val, dtype=torch.float32).unsqueeze(1)
    add_log_bg("🔄 开始模型训练分析...")
    model = CNNLSTM(len(feats), SEQ_WINDOW)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    best_val_loss = float('inf')
    patience = 0
    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()
        pred_up, pred_in = model(X_train_t)
        loss_train = criterion(pred_up, y_up_train_t) + criterion(pred_in, y_in_train_t)
        loss_train.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred_up_val, pred_in_val = model(X_val_t)
            loss_val = criterion(pred_up_val, y_up_val_t) + criterion(pred_in_val, y_in_val_t)
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            patience = 0
            torch.save(model.state_dict(), MODEL_FILE)
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    model.load_state_dict(torch.load(MODEL_FILE, map_location='cpu'))
    add_log_bg("✅ 量化模型分析完成\n---")

    df_result = df.iloc[SEQ_WINDOW - 1:].copy()
    model.eval()
    all_probs_up, all_probs_in = [], []
    with torch.no_grad():
        for i in range(len(df_result)):
            seq = feat_data[i:i + SEQ_WINDOW]
            seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)
            p_up, p_in = model(seq_t)
            all_probs_up.append(p_up.item())
            all_probs_in.append(p_in.item())
    df_result["model_prob_up"] = all_probs_up
    df_result["model_prob_in"] = all_probs_in
    df_result["indicator_score"] = df_result.apply(calc_indicator_score, axis=1)
    df_result["prob_up"] = df_result.apply(final_prob, axis=1)
    df_result["prob_in"] = df_result["model_prob_in"].clip(0.05, 0.95)
    df_result["资金状态"] = df_result.apply(lambda r: get_status(r["prob_in"], r["ret"], r["gap_up"], r["consec_up"]), axis=1)
    df_result["KDJ"] = df_result.apply(get_kdj, axis=1)
    df_result["RSI"] = df_result.apply(get_rsi, axis=1)
    df_result["MACD"] = df_result.apply(get_macd, axis=1)
    df_result["MA趋势"] = df_result.apply(get_ma, axis=1)
    df_result["EXPMA状态"] = df_result.apply(get_exp_status, axis=1)
    price_level = calc_price_levels(df_result, PRICE_LOOKBACK)
    latest = df_result.iloc[-1]

    # ========== 手机优化1：精简表头+pipe自适应表格 ==========
    add_log_bg("## 📊 近期行情与全指标状态")
    show_df = df_result.tail(10)[["date","ret","MA趋势","KDJ","RSI","MACD","EXPMA状态","资金状态","prob_up"]].copy()
    show_df["ret"] = show_df["ret"].apply(lambda x: f"{x:.2%}")
    show_df["prob_up"] = show_df["prob_up"].apply(lambda x: f"{x:.2%}")
    # 超短列名，手机竖屏无横向滚动
    show_df.columns = ["日期","涨跌","MA","KDJ","RSI","MACD","EXPMA","资金","明日胜率"]
    add_log_bg(show_df.to_markdown(index=False, tablefmt="pipe"))
    add_log_bg("---")

    # ========== 手机优化2：精简总结表格 ==========
    add_log_bg("## 📋 最新综合指标结论")
    add_log_bg(f"""
| 项目 | 详情 |
| ---- | ---- |
| 交易日 | {latest['date']} |
| 现价(元) | {price_level['close']:.2f} |
| MA | {latest['MA趋势']} |
| KDJ | {latest['KDJ']} |
| RSI | {latest['RSI']} |
| MACD | {latest['MACD']} |
| EXPMA | {latest['EXPMA状态']} |
| 资金 | {latest['资金状态']} |
| 综合分 | {latest['indicator_score']:.2f} |
| 明日胜率 | **{latest['prob_up']:.2%}** |
""")
    add_log_bg("---")

    # ========== 手机优化3：点位短句分行 ==========
    add_log_bg("## 🎯 动态买卖参考点位")
    add_log_bg(f"""
- 🟢 低吸价：{price_level['best_buy']} 元
- 🟡 支撑位：{price_level['support']} 元
- 🔴 短线压力：{price_level['pressure']} 元
- 🔵 长线压力：{price_level['pressure_long']} 元
- 🟠 止盈参考：{price_level['sell_ref']} 元
- ⛔ 止损价位：{price_level['stop_loss']} 元
""")
    add_log_bg("---")

    # ========== 手机优化4：大标题拆分换行，避免文字截断 ==========
    add_log_bg("# 📈综合研判")
    add_log_bg("### 概率+指标+EXPMA多周期+资金联合策略")
    prob = latest["prob_up"]
    fund_status = latest["资金状态"]
    kdj_status = latest["KDJ"]
    rsi_status = latest["RSI"]
    macd_status = latest["MACD"]
    ma_trend = latest["MA趋势"]
    exp_status = latest["EXPMA状态"]
    buy_price = price_level['best_buy']
    support = price_level['support']
    pressure = price_level['pressure']
    sell_ref = price_level['sell_ref']
    stop_loss = price_level['stop_loss']
    position_ratio = ""
    main_suggest = ""
    short_suggest = ""
    mid_suggest = ""
    long_suggest = ""
    risk_tip = ""

    # 原有仓位判断逻辑完全保留
    if prob > 0.60:
        if fund_status in ("🔥强势流入", "✅温和流入") and ma_trend == "📈多头" and latest["exp_bull"] == 1:
            position_ratio = "建议仓位：6~8成（积极做多）"
            main_suggest = "✅ 多方全面共振：上涨概率高、资金进场、均线+EXPMA多头排列，趋势明确"
            short_suggest = f"【短线1-3日】股价站稳5/10日线，可在{buy_price}低吸，不追高"
            mid_suggest = f"【波段5-15日】12/26日线多头延续，依托波段均线持有，{sell_ref}附近分批止盈"
            long_suggest = "【中长线】完整多头趋势，回调至20日线可加仓，耐心持有主升浪"
            risk_tip = f"风控：有效跌破{stop_loss}立即止损，杜绝深套"
        elif (kdj_status == "🟩超卖" or rsi_status == "🟩超卖") and macd_status == "✅金叉":
            position_ratio = "建议仓位：4~5成（抄底试仓）"
            main_suggest = "⚪ 低位反转信号：指标超卖+MACD金叉，看涨概率偏高，属于反弹行情"
            short_suggest = f"【短线】以反弹思路为主，{pressure}压力位附近减仓"
            mid_suggest = "【波段】反弹行情不恋战，见压力优先落袋为安"
            long_suggest = "【中长线】趋势未完全扭转，暂不重仓长线布局"
            risk_tip = f"风控：反弹不确定性较强，严格守住{stop_loss}止损线"
        elif kdj_status=="🟩超卖" and rsi_status=="🟩超卖":
            position_ratio = "建议仓位：3~4成（低位潜伏）"
            main_suggest = "⚪ 双指标低位超卖，技术性反弹概率大幅提升，潜伏布局"
            short_suggest = f"【短线】依托{support}低吸博弈反弹，压力{pressure}分批止盈"
            mid_suggest = "【波段】低位反转初期，小仓持有等反弹兑现"
            long_suggest = "【中长线】等待EXPMA拐头多头再加仓"
            risk_tip = f"风控：跌破{stop_loss}离场规避阴跌"
        else:
            position_ratio = "建议仓位：4~6成（谨慎做多）"
            main_suggest = "✅ 看涨概率较高，但局部指标存在分化，以波段思维为主"
            short_suggest = f"【短线】回踩{support}~{buy_price}区间低吸，压力位{pressure}减仓"
            mid_suggest = "【波段】持有至压力区逐步兑现，不盲目看高一线"
            long_suggest = "【中长线】趋势一般，控制仓位，不长期持仓"
            risk_tip = "风控：指标分化，警惕冲高回落风险"
    elif prob > 0.48:
        if kdj_status=="🟩超卖" or rsi_status=="🟩超卖":
            position_ratio = "建议仓位：2~3成（轻仓潜伏）"
            main_suggest = "⚪ 中性偏多+低位超卖，小仓博弈技术性反弹"
            short_suggest = f"【短线】最佳低吸介入价{buy_price}附近小仓试错，不重仓"
            mid_suggest = "【波段】快进快出，不格局波段行情"
            long_suggest = "【中长线】趋势震荡，放弃长线布局"
            risk_tip = "风控：盈亏比一般，单笔仓位不宜过重"
        else:
            position_ratio = "建议仓位：2~3成（轻仓试错）"
            main_suggest = "⚪ 中性偏多：涨跌概率接近，整体偏强但做多动能不足"
            short_suggest = f"【短线】仅在{buy_price}最佳低吸介入价参与，冲高不追"
            mid_suggest = "【波段】快进快出，不格局波段行情"
            long_suggest = "【中长线】趋势震荡，放弃长线布局"
            risk_tip = "风控：盈亏比一般，单笔仓位不宜过重"
    elif prob > 0.35:
        if kdj_status=="🟩超卖" and rsi_status=="🟩超卖":
            position_ratio = "建议仓位：2成（潜伏试错）"
            main_suggest = "⚠️ 整体震荡但双指标深度超卖，存在超跌反弹机会，小仓试错"
            short_suggest = f"【短线】回踩{buy_price}低吸，压力位止盈"
            mid_suggest = "【波段】反弹见压减仓，不长期持有"
            long_suggest = "【中长线】仍震荡，不重仓"
            risk_tip = "风控：震荡+低位，破止损果断离场"
        elif kdj_status=="🟩超卖" or rsi_status=="🟩超卖":
            position_ratio = "建议仓位：1~2成（极小仓观望试错）"
            main_suggest = "⚠️ 震荡格局，单一指标超卖，小仓搏反弹，不重仓"
            short_suggest = "【短线】只极端低位小仓参与"
            mid_suggest = "【波段】回避波段操作"
            long_suggest = "【中长线】持币等待方向选择"
            risk_tip = "风控：震荡行情来回打脸，小仓参与"
        else:
            position_ratio = "建议仓位：0成（观望）"
            main_suggest = "⚠️ 震荡格局：涨跌概率均衡，做多信号薄弱，观望为最优选择"
            short_suggest = "【短线】短线机会零散，不主动开仓"
            mid_suggest = "【波段】区间反复震荡，回避波段操作"
            long_suggest = "【中长线】趋势模糊，持币等待方向选择"
            risk_tip = "风控：震荡行情来回打脸，减少操作频率"
    else:
        if kdj_status=="🟩超卖" and rsi_status=="🟩超卖":
            position_ratio = "建议仓位：1~2成（超跌潜伏）"
            main_suggest = "❌ 整体偏空但双指标深度超跌，博弈超跌修复反弹，极小仓"
            short_suggest = f"【短线】仅{support}附近埋伏，破位止损"
            mid_suggest = "【波段】反弹即兑现，不恋战"
            long_suggest = "【中长线】空头趋势，绝不加仓"
            risk_tip = f"风控：下跌趋势，即使超卖也严控仓位，破{stop_loss}离场"
        else:
            position_ratio = "建议仓位：0成（空仓回避）"
            main_suggest = "❌ 空头格局：明日上涨概率偏低，资金与指标偏空，优先规避"
            if latest["exp_dead"] == 1 or latest["exp12"] < latest["exp26"]:
                short_suggest = "【短线】EXPMA波段死叉/空头，坚决不抄底、不抢反弹"
                mid_suggest = "【波段】波段趋势走弱，持仓逢高减仓"
                long_suggest = "【中长线】中长期趋势承压，耐心等待底部信号"
            else:
                short_suggest = "【短线】弱势震荡，反弹力度有限"
                mid_suggest = "【波段】无明确做多机会，持币为主"
                long_suggest = "【中长线】观望等待趋势反转"
            risk_tip = f"风控：下跌风险偏大，即使回踩{support}也不盲目接盘"

    add_log_bg(f"""
**📌 {position_ratio}**

> 📝 核心结论：{main_suggest}

- 🔹 **短线策略**：{short_suggest}
- 🔹 **波段策略**：{mid_suggest}
- 🔹 **中长线策略**：{long_suggest}
""")
    add_log_bg("### ⚠️ 风险提示")
    add_log_bg(f"🛡️ {risk_tip}")

    # 额外提醒
    tip_msg = ""
    if rsi_status == "🟥超买" or kdj_status == "🟥超买":
        tip_msg = "💡 额外提醒：当前KDJ/RSI处于超买区间，谨防高位回落，忌追涨！"
    elif rsi_status == "🟩超卖" and kdj_status == "🟩超卖":
        tip_msg = "💡 额外提醒：KDJ+RSI双重深度超卖，超跌反转概率大增，可低位分批潜伏"
    elif rsi_status == "🟩超卖" or kdj_status == "🟩超卖":
        tip_msg = "💡 额外提醒：当前KDJ/RSI处于超卖区间，存在技术性反弹机会，不宜过度杀跌"
    if tip_msg:
        add_log_bg(f"\n> {tip_msg}")

    add_log_bg("\n---\n✅ **本次量化分析结束**，随时录入新代码即可重新测算。\n> 温馨提醒：行情波动有风险，理性规划仓位，谨慎决策。")

    if os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)
    is_running = False

# ===================== Streamlit全局手机自适应配置 =====================
if __name__ == "__main__":
    # 手机自适应关键配置
    st.set_page_config(
        page_title="澄渊投策・智能量化研判系统",
        layout="wide",
        initial_sidebar_state="auto"
    )
    st.title("澄渊投策 · 智能量化研判系统")
    with st.sidebar:
        stock_input = st.text_input("输入6位股票代码", value="601138")
        check_wfo = st.checkbox("开启滚动分析(设备性能要求高)", value=True)
        run_btn = st.button("开始分析预测", type="primary")
    res_box = st.empty()
    log_content = []
    if run_btn:
        if is_running:
            res_box.warning("【温馨提示】模型正在多因子运算中，建模耗时较长，请耐心等待，不要重复操作")
        else:
            while not log_queue.empty():
                log_queue.get()
            log_content.clear()
            is_running = True
            t = threading.Thread(target=run_analysis, args=(stock_input, check_wfo), daemon=True)
            t.start()
    # 队列实时刷新
    while is_running or not log_queue.empty():
        try:
            msg = log_queue.get(timeout=0.05)
            log_content.append(msg)
            res_box.markdown("\n".join(log_content))
        except:
            continue
