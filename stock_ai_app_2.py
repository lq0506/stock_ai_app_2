import pandas as pd
import baostock as bs
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import streamlit as st

st.set_page_config(page_title="金融指标股票分析", layout="wide")

# 针对你截图里的两个类名，全方位封杀
hide_branding = """
<style>
/* 1. 针对右边皇冠按钮：.viewerBadge_nim44_23 */
._container_gzau3_1._viewerBadge_nim44_23,
a[class*="_viewerBadge_nim44_23"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    position: absolute !important;
    right: -9999px !important;
    bottom: -9999px !important;
    transform: scale(0) !important;
    z-index: -9999 !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
}

/* 2. 针对左边H图标：.profileContainer_gzau3_53 */
._profileContainer_gzau3_53,
div[class*="_profileContainer_gzau3_53"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    position: absolute !important;
    right: -9999px !important;
    bottom: -9999px !important;
    transform: scale(0) !important;
    z-index: -9999 !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
}

/* 3. 兜底：隐藏所有可能的父容器 */
div:has(> ._profileContainer_gzau3_53),
div:has(> ._container_gzau3_1._viewerBadge_nim44_23) {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* 4. 额外隐藏 Streamlit 自带的所有其他品牌元素 */
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stToolbar"] {
    visibility: hidden !important;
    display: none !important;
}
</style>

<script>
// 动态删除这两个元素，防止 CSS 被覆盖
function removeBadges() {
    // 删除皇冠按钮
    const badge = document.querySelector('._container_gzau3_1._viewerBadge_nim44_23');
    if (badge) badge.remove();

    // 删除H图标
    const profile = document.querySelector('._profileContainer_gzau3_53');
    if (profile) profile.remove();
}

// 页面加载后立即执行
window.addEventListener('load', removeBadges);
// 每隔100ms检查一次，防止元素被重新生成
setInterval(removeBadges, 100);
</script>
"""

st.markdown(hide_branding, unsafe_allow_html=True)


# ===================== 全局配置 =====================
SEQ_WINDOW = 15
today = datetime.today().strftime("%Y-%m-%d")

# ===================== 隐藏右上角菜单 + 隐藏底部水印 =====================
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def get_code(code):
    if len(code) != 6:
        return ""
    if code.startswith(("600", "601", "603", "605")):
        return f"sh.{code}"
    else:
        return f"sz.{code}"

# ===================== 指标计算函数 =====================
def calc_kdj(df, n=9, m1=3, m2=3):
    low_min = df["low"].rolling(n).min()
    high_max = df["high"].rolling(n).max()
    rsv = (df["close"] - low_min) / (high_max - low_min + 1e-6) * 100
    df["k"] = rsv.ewm(alpha=1/m1, adjust=False).mean()
    df["d"] = df["k"].ewm(alpha=1/m2, adjust=False).mean()
    df["j"] = 3 * df["k"] - 2 * df["d"]
    return df

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    rsi = 100 - 100 / (1 + rs)
    return rsi

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
        X.append(data[i:i+window])
        y_up.append(label_up[i + window - 1])
        y_in.append(label_in[i + window - 1])
    return np.array(X), np.array(y_up), np.array(y_in)

# ===================== CNN+LSTM模型 =====================
class CNNLSTM(nn.Module):
    def __init__(self, feat_dim, seq_len):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(feat_dim, 64, 3, padding=1), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(64, 32, 3, padding=1), nn.ReLU(), nn.Dropout(0.2)
        )
        self.lstm = nn.LSTM(32, 16, 2, batch_first=True, dropout=0.2)
        self.fc_up = nn.Linear(16,1)
        self.fc_in = nn.Linear(16,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.cnn(x).permute(0,2,1)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:,-1,:]
        return self.sigmoid(self.fc_up(out)), self.sigmoid(self.fc_in(out))

# ===================== 打分与约束函数 =====================
def calc_indicator_score(row):
    score = 0.0
    if row["ma5_above_ma20"] == 1: score +=15
    else: score -=15
    if row["price_above_ma20"] ==1: score +=10
    else: score -=10
    if row["price_above_ma5"] ==1: score +=5
    else: score -=5
    if row["kdj_over_sell"] ==1: score +=20
    elif row["kdj_over_buy"] ==1: score -=20
    if row["macd_gold"] ==1: score +=25
    elif row["macd_dead"] ==1: score -=25
    if row["rsi_over_sell"] ==1: score +=20
    elif row["rsi_over_buy"] ==1: score -=20
    if row["model_prob_in"] >0.5: score +=35
    elif row["model_prob_in"] <0.2: score -=35
    return np.clip(score/105, -1.0,1.0)

def hard_constraint(ret):
    if ret < -0.03: return 0.3
    elif ret < -0.01: return 0.45
    elif ret >0.03: return 0.85
    elif ret >0.01: return 0.75
    return 0.7

def final_prob(row):
    base = row["model_prob_up"]
    score = row["indicator_score"]
    ret = row["ret"]
    prob = base*0.4 + (0.5 + score*0.5)*0.6
    max_p = hard_constraint(ret)
    prob = min(prob, max_p)
    return float(np.clip(prob, 0.05, 0.95))

def get_status(p, ret, gap_up, consec_up):
    if ret < -0.02: return "💸资金流出"
    if consec_up >=2 and gap_up>2.5: return "🔥强势流入"
    if p>0.6: return "🔥强势流入"
    if p>0.35: return "✅温和流入"
    if p<0.2: return "💸资金流出"
    return "正常"

# ===================== 指标文字 =====================
def get_kdj(row):
    return "🟥超买" if row["kdj_over_buy"] else "🟩超卖" if row["kdj_over_sell"] else "正常"
def get_rsi(row):
    return "🟥超买" if row["rsi_over_buy"] else "🟩超卖" if row["rsi_over_sell"] else "正常"
def get_macd(row):
    return "✅金叉" if row["macd_gold"] else "❌死叉" if row["macd_dead"] else "中性"
def get_ma(row):
    return "📈多头" if row["ma5_above_ma20"] else "📉空头"

# ===================== 手机APP界面 =====================
def main():
    st.set_page_config(page_title="股票分析", page_icon="📈")
    st.title("📈 金融指标股票分析33333")
    st.divider()

    stock_code = st.text_input("请输入6位股票代码", value="603629")
    run_btn = st.button("开始分析预测")

    if run_btn:
        if len(stock_code)!=6:
            st.error("请输入正确6位股票代码！")
            return

        stock_sym = get_code(stock_code)
        progress_text = st.empty()
        progress_bar = st.progress(0)

        # 1.登录接口
        progress_text.text("正在获取股票行情...")
        progress_bar.progress(10)
        bs.login()

        start_date = "2023-01-01"
        end_date = today

        rs = bs.query_history_k_data_plus(
            code=stock_sym,
            fields="date,open,high,low,close,volume,amount,pctChg",
            start_date=start_date, end_date=end_date, frequency="d"
        )

        data = []
        while rs.error_code == '0' and rs.next():
            data.append(rs.get_row_data())
        df = pd.DataFrame(data, columns=rs.fields)
        bs.logout()

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        progress_text.text("正在获取历史K线数据...")
        progress_bar.progress(25)

        # 数据清洗
        num_cols = ["open","high","low","close","volume","amount","pctChg"]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna().reset_index(drop=True)

        # 指标计算
        progress_text.text("计算MA/KDJ/RSI/MACD/BOLL技术指标...")
        progress_bar.progress(55)

        df["prev_close"] = df["close"].shift(1)
        df["ret"] = df["close"].pct_change()
        df["gap_up"] = (df["open"] - df["prev_close"]) / df["prev_close"] *100
        df["intra_strength"] = (df["close"] - df["open"]) / (df["high"] - df["low"] +1e-6)

        df["vol_ma5"] = df["volume"].rolling(5).mean()
        df["vol_ratio"] = df["volume"] / df["vol_ma5"]
        df["amount_ma5"] = df["amount"].rolling(5).mean()
        df["amount_ratio"] = df["amount"] / df["amount_ma5"]

        df["up_day"] = (df["ret"]>0.015).astype(int)
        df["consec_up"] = df["up_day"].groupby((df["up_day"] != df["up_day"].shift(1))).cumsum()
        df["limit_up"] = (df["ret"]>0.09).astype(int)
        df["strong_up"] = (df["ret"]>0.02).astype(int)
        df["big_vol"] = (df["vol_ratio"]>1.1).astype(int)

        df["ma5"] = df["close"].rolling(5).mean()
        df["ma20"] = df["close"].rolling(20).mean()
        df["ma5_above_ma20"] = (df["ma5"]>df["ma20"]).astype(int)
        df["price_above_ma5"] = (df["close"]>df["ma5"]).astype(int)
        df["price_above_ma20"] = (df["close"]>df["ma20"]).astype(int)
        df["bias_ma5"] = (df["close"]-df["ma5"])/(df["ma5"]+1e-6)
        df["bias_ma20"] = (df["close"]-df["ma20"])/(df["ma20"]+1e-6)

        df = calc_kdj(df)
        df["kdj_over_buy"] = (df["j"]>85).astype(int)
        df["kdj_over_sell"] = (df["j"]<20).astype(int)

        df["rsi14"] = calc_rsi(df["close"])
        df["rsi_over_buy"] = (df["rsi14"]>70).astype(int)
        df["rsi_over_sell"] = (df["rsi14"]<30).astype(int)

        df = calc_macd(df)
        df["macd_gold"] = ((df["diff"]>df["dea"]) & (df["diff"].shift(1)<=df["dea"].shift(1))).astype(int)
        df["macd_dead"] = ((df["diff"]<df["dea"]) & (df["diff"].shift(1)>=df["dea"].shift(1))).astype(int)

        df["boll_mid"] = df["close"].rolling(20).mean()
        df["boll_std"] = df["close"].rolling(20).std()
        df["boll_up"] = df["boll_mid"] + 2*df["boll_std"]
        df["boll_down"] = df["boll_mid"] - 2*df["boll_std"]
        df["price_touch_up"] = (df["close"]>=df["boll_up"]).astype(int)
        df["price_touch_down"] = (df["close"]<=df["boll_down"]).astype(int)

        feats = [
            "gap_up","ret","intra_strength","vol_ratio","amount_ratio","consec_up","limit_up",
            "ma5_above_ma20","price_above_ma5","price_above_ma20","bias_ma5","bias_ma20",
            "k","d","j","kdj_over_buy","kdj_over_sell",
            "rsi14","rsi_over_buy","rsi_over_sell",
            "diff","dea","macd","macd_gold","macd_dead",
            "price_touch_up","price_touch_down","strong_up","big_vol"
        ]

        df = df.dropna(subset=feats).reset_index(drop=True)

        # 标签
        df["ret_tomorrow"] = df["ret"].shift(-1)
        df["label_up"] = (df["ret_tomorrow"]>0.005).astype(int)
        df["label_in"] = ((df["consec_up"]>=2)|(df["gap_up"]>3)|((df["ret"]>0.01)&(df["vol_ratio"]>1.05))).astype(int)

        # 模型
        progress_text.text("数据获取分析中...")
        progress_bar.progress(80)

        scaler = StandardScaler()
        feat_data = scaler.fit_transform(df[feats])
        X_seq, y_up_arr, y_in_arr = create_sequences(feat_data, df["label_up"].values, df["label_in"].values, SEQ_WINDOW)

        X_tensor = torch.tensor(X_seq, dtype=torch.float32).permute(0,2,1)
        y_up_tensor = torch.tensor(y_up_arr, dtype=torch.float32).unsqueeze(1)
        y_in_tensor = torch.tensor(y_in_arr, dtype=torch.float32).unsqueeze(1)

        model = CNNLSTM(len(feats), SEQ_WINDOW)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(300):
            opt.zero_grad()
            pred_up, pred_in = model(X_tensor)
            loss = nn.BCELoss()(pred_up, y_up_tensor) + nn.BCELoss()(pred_in, y_in_tensor)
            loss.backward()
            opt.step()

        # 预测
        model.eval()
        all_probs_up = []
        all_probs_in = []
        for i in range(len(feat_data)-SEQ_WINDOW+1):
            seq = feat_data[i:i+SEQ_WINDOW]
            seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).permute(0,2,1)
            p_up, p_in = model(seq)
            all_probs_up.append(p_up.item())
            all_probs_in.append(p_in.item())

        df_result = df.iloc[SEQ_WINDOW-1:].copy()
        df_result["model_prob_up"] = all_probs_up
        df_result["model_prob_in"] = all_probs_in

        df_result["indicator_score"] = df_result.apply(calc_indicator_score, axis=1)
        df_result["prob_up"] = df_result.apply(final_prob, axis=1)
        df_result["prob_in"] = df_result["model_prob_in"].clip(0.05,0.95)
        df_result["资金状态"] = df_result.apply(lambda r:get_status(r["prob_in"],r["ret"],r["gap_up"],r["consec_up"]),axis=1)

        # 生成指标文字
        df_result["KDJ"] = df_result.apply(get_kdj, axis=1)
        df_result["RSI"] = df_result.apply(get_rsi, axis=1)
        df_result["MACD"] = df_result.apply(get_macd, axis=1)
        df_result["MA趋势"] = df_result.apply(get_ma, axis=1)

        # ===================== 显示最近10天明细 =====================
        st.markdown("---")
        st.subheader("📋 最近10天历史明细")
        st.text("=" * 80)
        st.text(f"【{stock_sym} 五大指标+资金融合预测系统】")
        st.text("=" * 80)

        table_lines = []
        for _, row in df_result.tail(10).iterrows():
            line = (f"{row['date']} | 涨:{row['ret']:.2%} | MA:{row['MA趋势']} | KDJ:{row['KDJ']} | RSI:{row['RSI']} | MACD:{row['MACD']} | 资金:{row['资金状态']} | 明涨:{row['prob_up']:.2%}")
            table_lines.append(line)

        st.code("\n".join(table_lines))

        # ===================== 界面输出 =====================
        progress_text.text("✅ 分析完成！")
        progress_bar.progress(100)
        # st.success("所有指标预测完成")

        # 展示结果
        latest = df_result.iloc[-1]
        close = latest["close"]
        best_buy = round(close*0.97,2)
        support = round(close*0.95,2)
        pressure = round(close*1.05,2)
        stop = round(close*0.93,2)

        st.subheader("📊 最新行情综合结论")
        st.write(f"交易日期：{latest['date']}")
        st.write(f"当前股价：{close:.2f} 元")
        st.write(f"当日涨跌幅：{latest['ret']:.2%}")
        st.write(f"资金流向状态：{latest['资金状态']}")
        st.write(f"明日上涨综合概率：**{latest['prob_up']:.2%}**")

        st.subheader("💰 实战买卖价位")
        st.write(f"🟢 最佳低吸价：{best_buy} 元")
        st.write(f"🟡 强支撑价位：{support} 元")
        st.write(f"🔴 强压力价位：{pressure} 元")
        st.write(f"⛔ 止损参考价：{stop} 元")

        st.subheader("📌 操作建议")
        if latest["prob_up"]>0.55:
            st.info("✅ 综合看多：指标共振+资金流入，适合低吸布局")
        elif latest["prob_up"]<0.45:
            st.warning("❌ 综合看空：指标走弱资金流出，优先观望不进场")
        else:
            st.info("⚪ 市场震荡：多空分歧较大，等待趋势明确再操作")

if __name__ == "__main__":
    main()
