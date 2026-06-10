"""
Streamlit App — Bundle Recommendation System (Instacart).
Hiển thị gợi ý mua kèm từ các model: Ochiai, Item2Vec, DeepWalk, Ensemble + CB Filter.
Dữ liệu tiếng Việt từ data/processed/*_vi.csv

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

from src.config import MODEL_DIR, PROCESSED_DIR
from src.models.assoc_rules import AssocRulesModel
from src.models.cb_filter import CBFilter
from src.models.deepwalk import DeepWalkModel
from src.models.ensemble import EnsembleModel
from src.models.item2vec import Item2VecModel
from src.models.ochiai import OchiaiModel

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
def _read_csv_vi(filename):
    """Đọc CSV tiếng Việt dùng pandas (xử lý BOM, dấu phẩy trong tên, Unicode encoding)."""
    return pd.read_csv(filename, encoding="utf-8-sig")


@st.cache_data
def load_products_vi():
    """Load dữ liệu sản phẩm tiếng Việt + merge với products.parquet."""
    products = pd.read_parquet(os.path.join(PROCESSED_DIR, "products.parquet"))

    products_vi = _read_csv_vi(os.path.join(PROCESSED_DIR, "products_vi.csv"))
    aisles_vi = _read_csv_vi(os.path.join(PROCESSED_DIR, "aisles_vi.csv"))
    depts_vi = _read_csv_vi(os.path.join(PROCESSED_DIR, "departments_vi.csv"))

    # Ép kiểu product_id cho cả hai bảng để đồng bộ kiểu dữ liệu trước merge
    products["product_id"] = products["product_id"].astype(int)
    products_vi["product_id"] = products_vi["product_id"].astype(int)

    # Đồng bộ tương tự cho aisle và department
    products["aisle_id"] = products["aisle_id"].astype(int)
    aisles_vi["aisle_id"] = aisles_vi["aisle_id"].astype(int)

    products["department_id"] = products["department_id"].astype(int)
    depts_vi["department_id"] = depts_vi["department_id"].astype(int)

    # Merge lần lượt từng bảng
    products = products.merge(products_vi, on="product_id", how="left")
    products = products.merge(aisles_vi, on="aisle_id", how="left")
    products = products.merge(depts_vi, on="department_id", how="left")

    # Dùng tên tiếng Việt, fallback về tiếng Anh nếu chưa có bản dịch
    products["display_name"] = products["product_name_vi"].fillna(
        products["product_name"]
    )

    return products


@st.cache_resource
def load_models():
    """Load tất cả models đã train (chạy 1 lần, cache)."""
    models = {}

    with st.spinner("Đang load CB Filter..."):
        cb = CBFilter()
        cb.product_vectors = scipy.sparse.load_npz(
            os.path.join(MODEL_DIR, "cb_filter", "product_vectors.npz")
        )
        with open(
            os.path.join(MODEL_DIR, "cb_filter", "product_id_to_idx.json")
        ) as f:
            cb.product_id_to_idx = {int(k): v for k, v in json.load(f).items()}
    models["cb"] = cb

    with st.spinner("Đang load Ochiai..."):
        ochiai = OchiaiModel()
        ochiai.load(os.path.join(MODEL_DIR, "ochiai"))
    models["ochiai"] = ochiai

    with st.spinner("Đang load Item2Vec..."):
        i2v = Item2VecModel()
        i2v.load(os.path.join(MODEL_DIR, "item2vec"))
    models["i2v"] = i2v

    with st.spinner("Đang load DeepWalk..."):
        dw = DeepWalkModel()
        dw.load(os.path.join(MODEL_DIR, "deepwalk"))
    models["dw"] = dw

    with st.spinner("Đang load Association Rules..."):
        arm = AssocRulesModel()
        arm.load(os.path.join(MODEL_DIR, "assoc_rules"))
    models["arm"] = arm

    with st.spinner("Đang khởi tạo Ensemble..."):
        ensemble = EnsembleModel()
        ensemble.fit(ochiai, i2v, dw, cb)
    models["ensemble"] = ensemble

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
    recs = models["ochiai"].recommend(product_id, top_k=top_k)
    results["Ochiai"] = {"recs": recs, "time": time.time() - t0}

    t0 = time.time()
    recs = models["i2v"].recommend(product_id, top_k=top_k)
    results["Item2Vec"] = {"recs": recs, "time": time.time() - t0}

    t0 = time.time()
    recs = models["dw"].recommend(product_id, top_k=top_k)
    results["DeepWalk"] = {"recs": recs, "time": time.time() - t0}

    t0 = time.time()
    recs = models["arm"].recommend(product_id, top_k=top_k)
    results["AssocRules"] = {"recs": recs, "time": time.time() - t0}

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

    from src.features.vectorizer import cb_similarity

    similarities = cb_similarity(cb.product_vectors, idx_a, valid_indices)
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
    show_ochiai = st.sidebar.checkbox("Ochiai", value=True)
    show_i2v = st.sidebar.checkbox("Item2Vec", value=True)
    show_dw = st.sidebar.checkbox("DeepWalk", value=True)
    show_arm = st.sidebar.checkbox("Association Rules", value=True)
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
            <p style="margin:5px 0;"><b>Gian hàng:</b> {product_row.get('aisle_vi', product_row.get('aisle', '?'))}</p>
            <p style="margin:5px 0;"><b>Ngành hàng:</b> {product_row.get('department_vi', product_row.get('department', '?'))}</p>
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
    if show_ochiai:
        model_names.append("Ochiai")
    if show_i2v:
        model_names.append("Item2Vec")
    if show_dw:
        model_names.append("DeepWalk")
    if show_arm:
        model_names.append("AssocRules")
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
                            "Gian hàng": rrow.get(
                                "aisle_vi", rrow.get("aisle", "?")
                            ),
                            "Ngành hàng": rrow.get(
                                "department_vi", rrow.get("department", "?")
                            ),  # THÊM CỘT
                            "Điểm số": f"{score:.4f}",
                        }
                    )
                else:
                    rows.append(
                        {
                            "Xếp hạng": rank,
                            "ID": rid,
                            "Tên sản phẩm": "?",
                            "Gian hàng": "?",
                            "Ngành hàng": "?",  # THÊM CỘT
                            "Điểm số": f"{score:.4f}",
                        }
                    )

            df = pd.DataFrame(rows)
            st.dataframe(df, width="stretch", hide_index=True)

    # ---- CB Filter chi tiết ----
    if show_cb_detail and show_ensemble_cb:
        ensemble_recs = results.get("Ensemble + CB", {}).get("recs", [])
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
                        dept = rrow.get(
                            "department_vi", rrow.get("department", "?")
                        )  # THÊM CỘT
                    else:
                        name = "?"
                        dept = "?"

                    label = (
                        "❌ **Substitute**"
                        if sim >= 0.8
                        else "✅ Complementary"
                    )
                    rows.append(
                        {
                            "ID": cid,
                            "Tên sản phẩm": name,
                            "Ngành hàng": dept,  # THÊM CỘT
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
                    st.write(f"- [{pid}] {name}")
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
                    st.write(f"- [{pid}] {name}")
            else:
                st.write("*(Không có)*")


if __name__ == "__main__":
    main()