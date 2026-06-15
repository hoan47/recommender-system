"""
Streamlit App — Bundle Recommendation System (Instacart).
Hiển thị gợi ý mua kèm từ các model: Item-CF, Item2Vec, Metapath2Vec, Ensemble + CB Filter.
Dữ liệu tiếng Việt từ products.parquet (product_name đã là tiếng Việt sau 01_load_data.py)

Chạy: streamlit run scripts/08_streamlit_app.py
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.sparse
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODEL_DIR, PROCESSED_DIR, ENS_CB_THRESHOLD
from src.models.ensemble import EnsembleModel

# ============================================================
# CẤU HÌNH TRANG
# ============================================================
st.set_page_config(
    page_title="Bundle Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# HÀM LOAD DỮ LIỆU (CACHE)
# ============================================================


@st.cache_data
def load_products_vi():
    """Load sản phẩm từ products.parquet — product_name đã là tiếng Việt."""
    products = pd.read_parquet(os.path.join(PROCESSED_DIR, "products.parquet"))
    # product_name đã là tiếng Việt sau load_products(use_vietnamese=True)
    products["display_name"] = products["product_name"]
    return products


@st.cache_resource
def load_models():
    """Load tất cả models đã train (chạy 1 lần, cache)."""
    models = {}

    with st.spinner("Đang load Ensemble (Item-CF + Item2Vec + Metapath2Vec + CB Filter)..."):
        ensemble = EnsembleModel.load()
    models["ensemble"] = ensemble

    # Sub-models được truy xuất qua ensemble để lấy recommendations riêng
    models["item_cf"] = ensemble.item_cf
    models["i2v"] = ensemble.item2vec
    models["mw"] = ensemble.metapath2vec
    models["cb"] = ensemble.cb_filter

    return models


# ============================================================
# HÀM ENSEMBLE WRAPPER (vì recommend() không nhận top_k param)
# ============================================================


def recommend_ensemble_with_topk(ensemble, product_id, top_k, use_cb_filter):
    """Wrapper: set final_k + top_k tạm thời để lấy đúng số lượng gợi ý."""
    orig_final = ensemble.final_k
    orig_top = ensemble.top_k
    ensemble.final_k = top_k
    if top_k > ensemble.top_k:
        ensemble.top_k = top_k
    try:
        return ensemble.recommend(product_id, use_cb_filter=use_cb_filter)
    finally:
        ensemble.final_k = orig_final
        ensemble.top_k = orig_top


def get_all_recommendations(product_id, top_k, models):
    """Lấy gợi ý từ tất cả models."""
    results = {}

    t0 = time.time()
    recs = models["item_cf"].recommend(product_id, top_k=top_k)
    results["Item-CF"] = {"recs": recs, "time": time.time() - t0}

    t0 = time.time()
    recs = models["i2v"].recommend(product_id, top_k=top_k)
    results["Item2Vec"] = {"recs": recs, "time": time.time() - t0}

    t0 = time.time()
    recs = models["mw"].recommend(product_id, top_k=top_k)
    results["Metapath2Vec"] = {"recs": recs, "time": time.time() - t0}

    t0 = time.time()
    recs = recommend_ensemble_with_topk(
        models["ensemble"], product_id, top_k, use_cb_filter=False
    )
    results["Ensemble (w/o CB)"] = {"recs": recs, "time": time.time() - t0}

    t0 = time.time()
    recs = recommend_ensemble_with_topk(
        models["ensemble"], product_id, top_k, use_cb_filter=True
    )
    results["Ensemble + CB"] = {"recs": recs, "time": time.time() - t0}

    return results


def get_cb_detail(product_id, candidates, models):
    """Lấy chi tiết CB Filter cho từng candidate."""
    cb = models["cb"]
    if product_id not in cb.product_id_to_idx:
        return None

    idx_a = cb.product_id_to_idx[product_id]
    valid_candidates = []
    valid_indices = []
    for cid, _ in candidates:
        if cid in cb.product_id_to_idx:
            valid_candidates.append(cid)
            valid_indices.append(cb.product_id_to_idx[cid])

    if not valid_indices:
        return None

    from src.features.vectorizer import cb_ensemble_similarity

    similarities = cb_ensemble_similarity(
        cb.product_vectors_tfidf, cb.product_vectors_count,
        idx_a, valid_indices, alpha=cb.alpha,
    )
    return list(zip(valid_candidates, similarities))


# ============================================================
# MAIN APP
# ============================================================


def main():
    st.title("🛒 Hệ Thống Gợi Ý Mua Kèm (Bundle Recommendation)")
    st.markdown("### Instacart Market Basket Analysis")

    # Load dữ liệu
    with st.spinner("Đang tải dữ liệu sản phẩm..."):
        products = load_products_vi()

    st.success(f"✅ Đã tải {len(products)} sản phẩm")

    # Load models
    with st.spinner("Đang tải models (lần đầu có thể mất 10-20 giây)..."):
        models = load_models()

    st.success("✅ Đã tải tất cả models")

    # ========================================================
    # SIDEBAR
    # ========================================================
    st.sidebar.header("🔍 Tìm kiếm sản phẩm")

    search_method = st.sidebar.radio(
        "Chọn phương thức:", ["Nhập product_id", "Chọn từ danh sách"]
    )

    product_id = None
    if search_method == "Nhập product_id":
        pid_input = st.sidebar.text_input("Product ID:", value="1")
        try:
            product_id = int(pid_input)
        except ValueError:
            st.sidebar.error("Vui lòng nhập số")
            product_id = None
    else:
        options = products["product_id"].tolist()
        labels = [
            f"[{pid}] {name[:60]}..." if len(name) > 60 else f"[{pid}] {name}"
            for pid, name in zip(
                products["product_id"], products["display_name"]
            )
        ]
        selected = st.sidebar.selectbox(
            "Chọn sản phẩm:",
            options=range(len(options)),
            format_func=lambda i: labels[i],
        )
        product_id = options[selected]

    top_k = st.sidebar.slider(
        "Số lượng gợi ý:", min_value=5, max_value=50, value=10
    )

    st.sidebar.header("📊 Hiển thị model")
    show_item_cf = st.sidebar.checkbox("Item-CF", value=True)
    show_i2v = st.sidebar.checkbox("Item2Vec", value=True)
    show_mw = st.sidebar.checkbox("Metapath2Vec", value=True)
    show_ensemble_no_cb = st.sidebar.checkbox("Ensemble (w/o CB)", value=True)
    show_ensemble_cb = st.sidebar.checkbox("Ensemble + CB", value=True)

    st.sidebar.markdown("---")
    show_cb_detail = st.sidebar.checkbox("🎯 Hiển thị CB chi tiết", value=True)

    # ========================================================
    # MAIN
    # ========================================================

    if product_id is None:
        st.warning("Vui lòng chọn sản phẩm")
        return

    product_row = products[products["product_id"] == product_id]
    if product_row.empty:
        st.error(f"❌ Không tìm thấy sản phẩm ID = {product_id}")
        return

    product_row = product_row.iloc[0]

    # ---- Thẻ sản phẩm mục tiêu (Tối ưu hóa màu hiển thị theo Theme hệ thống) ----
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            f"""
        <div style="padding: 15px; 
                    border: 2px solid #4CAF50; 
                    border-radius: 10px; 
                    background-color: var(--secondary-background-color); 
                    color: var(--text-color);">
            <h3 style="margin:0; color: #4CAF50;">📦 SẢN PHẨM MỤC TIÊU</h3>
            <p style="margin:5px 0;"><b>ID:</b> {product_id}</p>
            <p style="margin:5px 0;"><b>Tên:</b> {product_row['display_name']}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ---- Lấy gợi ý ----
    with st.spinner("Đang tính toán gợi ý..."):
        results = get_all_recommendations(product_id, top_k, models)

    # ---- Bảng so sánh ----
    st.markdown("## 📊 So Sánh Các Model")

    model_names = []
    if show_item_cf:
        model_names.append("Item-CF")
    if show_i2v:
        model_names.append("Item2Vec")
    if show_mw:
        model_names.append("Metapath2Vec")
    if show_ensemble_no_cb:
        model_names.append("Ensemble (w/o CB)")
    if show_ensemble_cb:
        model_names.append("Ensemble + CB")

    tabs = st.tabs(model_names)

    for tab_idx, model_name in enumerate(model_names):
        with tabs[tab_idx]:
            data = results.get(model_name)
            if data is None or not data["recs"]:
                st.warning(f"❌ Model {model_name} không có gợi ý.")
                continue

            recs = data["recs"]
            elapsed = data["time"]

            st.caption(f"⏱ Thời gian: {elapsed:.3f}s")

            rows = []
            for rank, (rid, score) in enumerate(recs, 1):
                rrow = products[products["product_id"] == rid]
                if not rrow.empty:
                    rrow = rrow.iloc[0]
                    rows.append(
                        {
                            "Xếp hạng": rank,
                            "ID": rid,
                            "Tên sản phẩm": rrow["display_name"],
                            "Điểm số": f"{score:.4f}",
                        }
                    )
                else:
                    rows.append(
                        {
                            "Xếp hạng": rank,
                            "ID": rid,
                            "Tên sản phẩm": "?",
                            "Điểm số": f"{score:.4f}",
                        }
                    )

            df = pd.DataFrame(rows)
            st.dataframe(df, width="stretch", hide_index=True)

    # ---- CB Filter chi tiết ----
    if show_cb_detail and show_ensemble_cb:
        # Lấy candidate từ kết quả gốc trước CB filter để hiển thị CB similarity chi tiết
        ensemble_recs = results.get("Ensemble (w/o CB)", {}).get("recs", [])
        if ensemble_recs:
            st.markdown("---")
            st.markdown("## 🎯 CB Filter Chi Tiết (Ensemble + CB)")

            detail = get_cb_detail(product_id, ensemble_recs, models)
            if detail:
                rows = []
                for cid, sim in detail:
                    rrow = products[products["product_id"] == cid]
                    if not rrow.empty:
                        rrow = rrow.iloc[0]
                        name = rrow["display_name"]
                    else:
                        name = "?"

                    label = (
                        "❌ **Substitute**"
                        if sim >= ENS_CB_THRESHOLD
                        else "✅ Complementary"
                    )
                    rows.append(
                        {
                            "ID": cid,
                            "Tên sản phẩm": name,
                            "CB Similarity": f"{sim:.4f}",
                            "Kết luận": label,
                        }
                    )

                df_detail = pd.DataFrame(rows)
                st.dataframe(df_detail, width="stretch", hide_index=True)
            else:
                st.info("ℹ️ Không có dữ liệu CB Filter cho sản phẩm này.")

    # ---- So sánh có/không CB ----
    if show_ensemble_cb and show_ensemble_no_cb:
        st.markdown("---")
        st.markdown("## 🔄 So Sánh Ensemble: Có CB vs Không CB")

        recs_cb = results.get("Ensemble + CB", {}).get("recs", [])
        recs_no_cb = results.get("Ensemble (w/o CB)", {}).get("recs", [])

        set_cb = set(pid for pid, _ in recs_cb)
        set_no_cb = set(pid for pid, _ in recs_no_cb)

        only_in_no_cb = set_no_cb - set_cb
        only_in_cb = set_cb - set_no_cb

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                "**❌ Bị CB Filter loại (trong w/o CB nhưng không có trong +CB):**"
            )
            if only_in_no_cb:
                for pid in only_in_no_cb:
                    rrow = products[products["product_id"] == pid]
                    name = (
                        rrow.iloc[0]["display_name"] if not rrow.empty else "?"
                    )
                    name_en = (
                        rrow.iloc[0]["product_name"] if not rrow.empty else "?"
                    )
                    st.write(f"- [{pid}] {name} ({name_en})")
            else:
                st.write("*(Không có)*")

        with col2:
            st.markdown("**✅ Chỉ xuất hiện trong Ensemble + CB:**")
            if only_in_cb:
                for pid in only_in_cb:
                    rrow = products[products["product_id"] == pid]
                    name = (
                        rrow.iloc[0]["display_name"] if not rrow.empty else "?"
                    )
                    name_en = (
                        rrow.iloc[0]["product_name"] if not rrow.empty else "?"
                    )
                    st.write(f"- [{pid}] {name} ({name_en})")
            else:
                st.write("*(Không có)*")


if __name__ == "__main__":
    main()