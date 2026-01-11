# ==============================================================================
# ファイル名: app.py
# 概要: YLS再犯予測ツール (Streamlit版)
# ==============================================================================

import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1. アプリの基本設定
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="YLS再犯予測テーブル",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 日本語フォント対応
import matplotlib
try:
    import japanize_matplotlib
except ImportError:
    pass 

# タイトル
st.title("🛡️ YLS Prediciton Table")

# ------------------------------------------------------------------------------
# 2. YLSデータの定義（Excel完全準拠）
# ------------------------------------------------------------------------------
YLS_DOMAINS = {
    "1. 非行歴": [
        {"id": "YLS_1a", "label": "a. 過去の家裁係属３回以上"},
        {"id": "YLS_1b", "label": "b. ２回以上の遵守事項違反"},
        {"id": "YLS_1c", "label": "c. 保護観察歴あり"},
        {"id": "YLS_1d", "label": "d. 過去に施設経験あり"},
        {"id": "YLS_1e", "label": "e. 現在３つ以上の事件が係属中"},
    ],
    "2. 家庭・養育": [
        {"id": "YLS_2a", "label": "a. 不十分な監護"},
        {"id": "YLS_2b", "label": "b. 子供を統制するのが困難"},
        {"id": "YLS_2c", "label": "c. 不適切なしつけ"},
        {"id": "YLS_2d", "label": "d. 一貫性を欠いた養育態度"},
        {"id": "YLS_2e", "label": "e. 父子間の劣悪な関係"},
        {"id": "YLS_2f", "label": "f. 母子間の劣悪な関係"},
    ],
    "3. 教育・雇用": [
        {"id": "YLS_3a", "label": "a. 教室での破壊的行動"},
        {"id": "YLS_3b", "label": "b. 学校での破壊的行動"},
        {"id": "YLS_3c", "label": "c. 成績不良"},
        {"id": "YLS_3d", "label": "d. 仲間関係の問題"},
        {"id": "YLS_3e", "label": "e. 対教師の問題"},
        {"id": "YLS_3f", "label": "f. 無断欠席"},
        {"id": "YLS_3g", "label": "g. 不就労で仕事を探していない"},
    ],
    "4. 仲間関係": [
        {"id": "YLS_4a", "label": "a. 非行をしている知り合いがいる"},
        {"id": "YLS_4b", "label": "b. 非行をしている友達がいる"},
        {"id": "YLS_4c", "label": "c. 健全な知り合いがほとんどいない"},
        {"id": "YLS_4d", "label": "d. 健全な友達がほとんどいない"},
    ],
    "5. 物質乱用": [
        {"id": "YLS_5a", "label": "a. 時々薬物を使用"},
        {"id": "YLS_5b", "label": "b. 薬物を常習"},
        {"id": "YLS_5c", "label": "c. アルコールを常習"},
        {"id": "YLS_5d", "label": "d. 物質乱用が社会生活を阻害している"},
        {"id": "YLS_5e", "label": "e. 物質の使用が犯罪に結びついている"},
    ],
    "6. 余暇娯楽": [
        {"id": "YLS_6a", "label": "a. 集団活動への不参加"},
        {"id": "YLS_6b", "label": "b. 有意義に時間を過ごしていない"},
        {"id": "YLS_6c", "label": "c. 興味関心の乏しさ"},
    ],
    "7. 人格行動": [
        {"id": "YLS_7a", "label": "a. 誇大な自尊心"},
        {"id": "YLS_7b", "label": "b. 身体的な攻撃性"},
        {"id": "YLS_7c", "label": "c. 癇癪を起こす"},
        {"id": "YLS_7d", "label": "d. 注意力の乏しさ"},
        {"id": "YLS_7e", "label": "e. 欲求不満耐性の乏しさ"},
        {"id": "YLS_7f", "label": "f. 罪悪感の乏しさ"},
        {"id": "YLS_7g", "label": "g. 言語的な攻撃性，無作法"},
    ],
    "8. 態度・志向": [
        {"id": "YLS_8a", "label": "a. 反社会的な態度・犯罪への志向"},
        {"id": "YLS_8b", "label": "b. 援助を求めない"},
        {"id": "YLS_8c", "label": "c. 強く援助を拒絶する"},
        {"id": "YLS_8d", "label": "d. 権威への反発，無視"},
        {"id": "YLS_8e", "label": "e. 他者への無関心，無感覚"},
    ]
}

ALL_FEATURES = []
JP_LABELS = {}
for items in YLS_DOMAINS.values():
    for item in items:
        ALL_FEATURES.append(item["id"])
        JP_LABELS[item["id"]] = item["label"]

# ------------------------------------------------------------------------------
# 3. モデルの読み込み (.ubj対応)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_ai_model():
    model = xgb.XGBClassifier()
    try:
        # ★軽量化モデル(.ubj)を読み込みます
        model.load_model("yls_model.ubj")
    except Exception as e:
        return None, f"モデル読み込みエラー: {e}"
    
    explainer = shap.TreeExplainer(model)
    return model, explainer

model, explainer = load_ai_model()

if model is None:
    st.error("⚠️ エラー: 'yls_model.ubj' が見つかりません。")
    st.stop()

# ------------------------------------------------------------------------------
# 4. 画面レイアウト（左35% : 右65%）
# ------------------------------------------------------------------------------
col_input, col_result = st.columns([0.35, 0.65])

# --- 左側：チェックボックス ---
user_inputs = {}

with col_input:
    st.markdown("### 📋 項目チェック")
    
    # 領域ごとに表示
    for domain_name, items in YLS_DOMAINS.items():
        # 青い縦線で見出しを表示
        st.markdown(
            f"<div style='border-left: 5px solid #007bff; padding-left: 8px; margin-top: 15px; font-weight: bold;'>{domain_name}</div>",
            unsafe_allow_html=True
        )
        
        for item in items:
            # チェックボックス
            is_checked = st.checkbox(item["label"], key=item["id"])
            user_inputs[item["id"]] = 1 if is_checked else 0

# --- 右側：結果表示 ---
with col_result:
    st.markdown("### 📊 分析結果")

    # データ作成
    input_df = pd.DataFrame([user_inputs])
    valid_input_df = input_df[ALL_FEATURES]

    # 予測
    prob = model.predict_proba(valid_input_df)[0][1]
    total_score = valid_input_df.sum(axis=1).values[0]

    # 色分けロジック
    if prob >= 0.71:
        color = "#dc3545" # Red
        text = "高リスク (High)"
        bg = "#ffe6e6"
    elif prob >= 0.33:
        color = "#fd7e14" # Orange
        text = "中リスク (Medium)"
        bg = "#fff3cd"
    elif prob >= 0.18:
        color = "#0d6efd" # Blue
        text = "低リスク (Low)"
        bg = "#e7f1ff"
    else:
        color = "#198754" # Green
        text = "最低リスク (Lowest)"
        bg = "#e8f5e9"

    # HTMLで結果表示 (文字サイズ調整版 24px)
    st.markdown(f"""
    <div style="
        border: 2px solid {color};
        border-radius: 8px;
        background-color: {bg};
        padding: 15px;
        text-align: center;
        margin-bottom: 20px;
    ">
        <div style="color:{color}; font-weight:bold; font-size:18px; margin-bottom:5px;">{text}</div>
        <hr style="border-top: 1px solid {color}; margin: 5px 0;">
        <div style="font-size: 12px; color: #555;">予想される再犯確率</div>
        <div style="font-size: 24px; font-weight: bold; color: {color}; line-height: 1.2;">
            {prob * 100:.1f}%
        </div>
        <p style="font-size: 14px; color: #333; margin-top: 5px;">
            合計得点: <b>{int(total_score)}</b> / 42点
        </p>
    </div>
    """, unsafe_allow_html=True)

    # SHAPグラフ表示
    st.markdown("**【要因分析 (全項目)】**")
    
    shap_values = explainer(valid_input_df)
    shap_values.feature_names = [JP_LABELS.get(f, f) for f in ALL_FEATURES]

    # グラフサイズ調整 (縦長・横広)
    fig, ax = plt.subplots(figsize=(10, 12))
    
    try:
        # 全項目表示 (max_display=42)
        shap.plots.waterfall(shap_values[0, :, 1], max_display=42, show=False)
    except:
        shap.plots.waterfall(shap_values[0], max_display=42, show=False)
    
    st.pyplot(fig)