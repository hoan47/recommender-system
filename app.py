import streamlit as st
import pandas as pd
import data_loader as dl
import dept_direction as dd
import models
import evaluate as ev
from config import PATH_OUTPUT_CSV

st.set_page_config(page_title="Product Recommendation System", layout="wide")


@st.cache_resource(show_spinner=False)
def init_system():
    dl.load_all()
    dd.build()
    models.build_all()
    return True


st.title("🛒 Product Recommendation System")
st.markdown("Select a seed product and compare recommendations across 4 models.")

with st.spinner("Loading data and building models (first run only)..."):
    init_system()

st.success("System ready.")

tab1, tab2 = st.tabs(["🔍 Explore Recommendations", "📊 Evaluate Models"])

with tab1:
    st.sidebar.header("Settings")
    product_options = [
        f"{pid} - {dl.name_map.get(pid, 'Unknown')}"
        for pid in dl.frequent_items
    ]

    selected_product = st.sidebar.selectbox(
        "Seed Product:",
        options=product_options,
        index=0,
    )

    top_k = st.sidebar.slider(
        "Top K recommendations:",
        min_value=5, max_value=50, value=10, step=5,
    )

    if selected_product:
        seed_pid  = int(selected_product.split(" - ")[0])
        seed_dept = dl.dept_name.get(dl.prod_dept_map.get(seed_pid, -1), "Unknown")

        st.markdown(
            f"### Viewing: **{selected_product.split(' - ')[1]}** "
            f"(ID: {seed_pid} | Dept: {seed_dept})"
        )
        st.divider()

        with st.spinner("Computing recommendations..."):
            recs_cb  = models.content_based_rec(seed_pid, k=top_k)
            recs_cf  = models.collab_rec(seed_pid, k=top_k)
            recs_kg  = models.kg_rec(seed_pid, k=top_k)
            recs_hyb = models.hybrid_rec(seed_pid, k=top_k)

        def render_recs(recs: list, title: str) -> None:
            st.subheader(title)
            if not recs:
                st.info("No recommendations available.")
                return
            for i, pid in enumerate(recs):
                pname = dl.name_map.get(pid, "Unknown")
                dept  = dl.dept_name.get(dl.prod_dept_map.get(pid, -1), "Unknown")
                st.markdown(
                    f"**{i+1}.** {pname}  \n"
                    f"<span style='color:gray; font-size:14px'>ID: {pid} | Dept: {dept}</span>",
                    unsafe_allow_html=True,
                )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_recs(recs_cb, "📝 Content-Based")
        with col2:
            render_recs(recs_cf, "🤝 Collaborative")
        with col3:
            render_recs(recs_kg, "🕸️ Knowledge Graph")
        with col4:
            render_recs(recs_hyb, "✨ Hybrid")

with tab2:
    st.header("📊 Model Evaluation")
    st.markdown(
        f"Evaluation on **{len(dl.test_cases):,}** test cases (user-based 20% split)."
    )

    if st.button("🚀 Run Evaluation", type="primary"):
        with st.spinner("Evaluating models..."):
            models_dict = {
                "Content-Based":        models.content_based_rec,
                "Collaborative (CF)":   models.collab_rec,
                "Knowledge Graph (KG)": models.kg_rec,
                "Hybrid (CB+CF+KG)":    models.hybrid_rec,
            }
            df_results = ev.run_comparison(models_dict, ks=(10, 50, 100))

        st.success("Evaluation complete.")
        st.subheader("🏆 Results")
        st.dataframe(df_results.style.format("{:.4f}"))

        st.subheader("⭐ Best Model per Metric")
        ks = (10, 50, 100)
        metrics = []
        for k in ks:
            metrics.extend([f"P@{k}", f"R@{k}", f"F1@{k}", f"H@{k}"])

        best_rows = []
        for m in metrics:
            best     = df_results[m].idxmax()
            best_rows.append({"Metric": m, "Best Model": best, "Score": f"{df_results.loc[best, m]:.4f}"})

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**K = 10**")
            st.table(pd.DataFrame([r for r in best_rows if "10" in r["Metric"] and "100" not in r["Metric"]]))
        with c2:
            st.markdown("**K = 50**")
            st.table(pd.DataFrame([r for r in best_rows if "50" in r["Metric"]]))
        with c3:
            st.markdown("**K = 100**")
            st.table(pd.DataFrame([r for r in best_rows if "100" in r["Metric"]]))

        st.info(f"Results saved to: `{PATH_OUTPUT_CSV}`")
