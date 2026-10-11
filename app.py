
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==========================================
# BUYER SEGMENTATION & INVESTMENT PROFILING
# REAL ESTATE MARKET INTELLIGENCE
# ==========================================

st.set_page_config(
    page_title="Real Estate Buyer Segmentation AI",
    page_icon="🏡",
    layout="wide"
)

st.title("🏡 Real Estate Buyer Segmentation AI")
st.subheader("Buyer Segmentation & Investment Profiling")
st.write(
    "Analyze customer characteristics, identify buyer segments, "
    "and explore real estate investment profiles using K-Means clustering."
)

# ------------------------------------------
# LOAD DATASET
# ------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("clients.csv")

try:
    original_df = load_data()
except FileNotFoundError:
    st.error(
        "clients.csv was not found. Make sure it is in the "
        "same main folder as app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Unable to load the dataset: {e}")
    st.stop()

st.success("Dataset loaded successfully!")

# Keep original data for download and display
result_df = original_df.copy()

# ------------------------------------------
# DATASET OVERVIEW
# ------------------------------------------
st.header("1. Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Buyers", f"{len(original_df):,}")
col2.metric("Total Original Features", len(original_df.columns))
col3.metric(
    "Missing Values",
    f"{int(original_df.isnull().sum().sum()):,}"
)

with st.expander("Preview Dataset"):
    st.dataframe(original_df.head(20), use_container_width=True)

# ------------------------------------------
# DATA CLEANING
# ------------------------------------------
df = original_df.copy()

df = df.drop(
    columns=["client_id", "first_name", "last_name"],
    errors="ignore"
)

if "date_of_birth" in df.columns:
    df["date_of_birth"] = pd.to_datetime(
        df["date_of_birth"], errors="coerce"
    )
    df["Age"] = 2025 - df["date_of_birth"].dt.year
    df = df.drop(columns=["date_of_birth"])

# Fill missing numeric values
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill missing categorical values
cat_cols = df.select_dtypes(include="object").columns
df[cat_cols] = df[cat_cols].fillna("Unknown")

# Encode categorical features
df = pd.get_dummies(df, drop_first=True)
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

if df.shape[1] < 1:
    st.error("No usable features were found in the dataset.")
    st.stop()

if len(df) < 4:
    st.error("At least four buyer records are needed for four clusters.")
    st.stop()

# ------------------------------------------
# FEATURE SCALING
# ------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ------------------------------------------
# ELBOW METHOD
# ------------------------------------------
st.header("2. Optimal Cluster Analysis")
st.write(
    "The Elbow Method compares clustering inertia for different "
    "cluster counts. Use the curve to explore a suitable cluster number."
)

max_k = min(9, len(df) - 1)
k_values = list(range(2, max_k + 1))
inertia = []

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    model.fit(X_scaled)
    inertia.append(model.inertia_)

fig_elbow, ax = plt.subplots(figsize=(8, 4))
ax.plot(k_values, inertia, marker="o")
ax.set_title("Elbow Method")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Inertia")
ax.grid(alpha=0.3)
st.pyplot(fig_elbow)
plt.close(fig_elbow)

# ------------------------------------------
# SELECT NUMBER OF CLUSTERS
# ------------------------------------------
st.header("3. Buyer Segmentation")

n_clusters = st.slider(
    "Select number of buyer segments",
    min_value=2,
    max_value=max_k,
    value=min(4, max_k)
)

kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)
df["Cluster"] = clusters

result_df["Cluster"] = clusters

# ------------------------------------------
# MODEL PERFORMANCE
# ------------------------------------------
score = silhouette_score(X_scaled, clusters)

col1, col2, col3 = st.columns(3)

col1.metric("Buyer Segments", n_clusters)
col2.metric("Silhouette Score", f"{score:.3f}")
col3.metric("Buyers Analyzed", f"{len(df):,}")

st.caption(
    "Silhouette scores range from -1 to 1. Higher scores generally "
    "indicate better-separated clusters. Interpret the score alongside "
    "the segment profiles."
)

# ------------------------------------------
# CLUSTER SUMMARY
# ------------------------------------------
st.header("4. Buyer Segment Summary")

cluster_summary = df.groupby("Cluster").mean(numeric_only=True)

st.dataframe(
    cluster_summary.round(3),
    use_container_width=True
)

# Buyer count per cluster
cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
    .rename_axis("Cluster")
    .reset_index(name="Number of Buyers")
)

st.subheader("Buyer Distribution")

fig_dist, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(
    cluster_counts["Cluster"].astype(str),
    cluster_counts["Number of Buyers"]
)
ax.set_title("Cluster Distribution")
ax.set_xlabel("Buyer Segment")
ax.set_ylabel("Number of Buyers")
ax.bar_label(bars)
ax.grid(axis="y", alpha=0.25)
st.pyplot(fig_dist)
plt.close(fig_dist)

# ------------------------------------------
# BUYER SEGMENT VISUALIZATION
# ------------------------------------------
st.header("5. Buyer Segments Visualization")

if X_scaled.shape[1] >= 2:
    fig_scatter, ax = plt.subplots(figsize=(8, 5))

    scatter = ax.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=clusters,
        cmap="Set1",
        alpha=0.7
    )

    ax.set_title("Buyer Segments Visualization")
    ax.set_xlabel(f"Scaled Feature: {df.columns[0]}")
    ax.set_ylabel(f"Scaled Feature: {df.columns[1]}")
    ax.legend(
        *scatter.legend_elements(),
        title="Cluster"
    )
    ax.grid(alpha=0.2)

    st.pyplot(fig_scatter)
    plt.close(fig_scatter)
else:
    st.info("At least two features are needed for the scatter plot.")

# ------------------------------------------
# INDIVIDUAL CLUSTER EXPLORER
# ------------------------------------------
st.header("6. Explore Individual Buyer Segments")

selected_cluster = st.selectbox(
    "Choose a buyer segment",
    options=sorted(df["Cluster"].unique()),
    format_func=lambda x: f"Buyer Segment {x}"
)

segment_data = result_df[result_df["Cluster"] == selected_cluster]

st.write(f"**Buyers in selected segment:** {len(segment_data):,}")
st.dataframe(segment_data.head(100), use_container_width=True)

# ------------------------------------------
# DOWNLOAD RESULTS
# ------------------------------------------
st.header("7. Download Analysis Results")

csv_data = result_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Buyer Segmentation Results (CSV)",
    data=csv_data,
    file_name="buyer_segmentation_results.csv",
    mime="text/csv"
)

st.caption(
    "Buyer segments are algorithm-generated groups. Cluster numbers "
    "are labels, not rankings or confirmed investment recommendations."
)

