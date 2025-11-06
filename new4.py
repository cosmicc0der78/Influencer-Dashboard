# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import networkx as nx
import plotly.graph_objects as go
from matplotlib import font_manager

# --- Page Setup ---
st.set_page_config(
    page_title="Influencer Graph Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Theme (dark sidebar, soft main, navy buttons) ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

        /* Main App (soft white background) */
        .stApp {
            background: #f1f5f9; /* slightly softer than pure white */
            font-family: 'IBM Plex Sans', sans-serif;
            color: #0f172a;
        }
        [data-testid="stAppViewContainer"] > .main {
            background: #f1f5f9;
        }

        /* Sidebar: deep navy */
        [data-testid="stSidebar"] {
            background: #0f172a !important;
            color: #f8fafc !important;
            border-right: 1px solid #1e293b !important;
            box-shadow: 2px 0 8px rgba(0, 0, 0, 0.12);
        }
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] label, 
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] span {
            color: #f8fafc !important;
            font-family: 'IBM Plex Sans', sans-serif;
        }

        /* Sidebar inputs */
        [data-testid="stSidebar"] input, 
        [data-testid="stSidebar"] .stTextInput input,
        [data-testid="stSidebar"] .stSelectbox > div > div {
            background: #0b1220 !important;
            color: #f8fafc !important;
            border: 1px solid #243045 !important;
            border-radius: 6px !important;
        }
        [data-testid="stSidebar"] input:focus {
            border-color: #324a66 !important;
            box-shadow: 0 0 0 1px #324a66 !important;
        }

        /* Buttons (navy-black) */
        .stDownloadButton button, .stButton>button {
            background: #0f172a !important;  /* navy-black */
            color: #ffffff !important;
            border: none !important;
            border-radius: 6px !important;
            padding: 10px 18px !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            box-shadow: 0 1px 4px rgba(0,0,0,0.12) !important;
            transition: all 0.18s ease-in-out !important;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }

        /* Ensure text/icons inside buttons are visible */
        .stButton>button *, .stDownloadButton button * {
            color: #ffffff !important;
            fill: #ffffff !important;
            text-shadow: none !important;
            opacity: 1 !important;
        }

        /* Hover effect */
        .stDownloadButton button:hover, .stButton>button:hover {
            background: #1e293b !important; /* slightly lighter on hover */
            color: #ffffff !important;
            transform: translateY(-1px);
            box-shadow: 0 6px 14px rgba(0,0,0,0.12) !important;
        }


        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background: #ffffff;
            border-radius: 8px;
            padding: 6px;
            border: 1px solid #e2e8f0;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 6px;
            color: #374151 !important;  /* normal text color */
            font-weight: 600;
            padding: 8px 18px;
            font-size: 0.95rem;
        }
        .stTabs [aria-selected="true"],
        .stTabs [aria-selected="true"] *{
            background-color: #0f172a !important;  /* dark background */
            color: #ffffff !important;             /* white text */
            fill: #ffffff !important;        /* ensures icons also turn white */
            text-shadow: none !important;    /* remove invisible glow from theme */
            opacity: 1 !important;           /* fix any dimming */
            font-weight: 700 !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15);
        }
        /* Optional: hover effect */
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #f1f5f9;
            color: #0f172a;
        }

        /* Metric containers */
        div[data-testid="metric-container"] {
            background: #ffffff !important;
            border-radius: 10px !important;
            padding: 18px !important;
            border: 1px solid #e6eef8 !important;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
        }
        [data-testid="stMetricLabel"] {
            color: #64748b !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }
        [data-testid="stMetricValue"] {
            color: #0f172a !important;
            font-weight: 700 !important;
        }

        /* DataFrame */
        [data-testid="stDataFrame"] {
            background: #ffffff !important;
            border-radius: 8px !important;
            border: 1px solid #e6eef8 !important;
        }

        /* Plotly containers */
        .js-plotly-plot {
            background: #ffffff !important;
            border-radius: 8px !important;
            padding: 14px !important;
            border: 1px solid #e6eef8 !important;
            box-shadow: 0 2px 8px rgba(15,23,42,0.04) !important;
        }

        /* Headings + text color on main panel */
        h1, h2, h3, p, .stMarkdown {
            color: #0f172a !important;
        }

        /* Minor */
        hr {
            border-color: #e6eef8 !important;
        }
        #MainMenu {visibility: visible;}
        footer {visibility: visible !important;}
        footer::before {
            content: "© 2025 Indu Sree N. | Crafted with caffeine, passion, and just the right amount of overthinking ☕💡 | Powered by Streamlit ⚡";
            display: block;
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: #1e293b;
            color: #cbd5e1;
            text-align: center;
            padding: 10px 5px;
            font-size: 0.9rem;
            font-family: 'IBM Plex Sans', sans-serif;
            z-index: 9999;
            border-top: 1px solid #334155;
        }
    </style>
""", unsafe_allow_html=True)

# --- Matplotlib / Seaborn default colors for consistent look ---
sns.set_style("whitegrid")  # activates Seaborn’s whitegrid style safely
plt.rcParams.update({
    "axes.facecolor": "#f8fafc",
    "axes.edgecolor": "#94a3b8",
    "axes.labelcolor": "#0f172a",
    "xtick.color": "#0f172a",
    "ytick.color": "#0f172a",
    "text.color": "#0f172a",
    "grid.color": "#94a3b8",
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
})
plt.rcParams.update({
    "font.size": 8,           # base font size (smaller)
    "axes.titlesize": 10,     # chart titles
    "axes.labelsize": 8,      # x/y labels
    "xtick.labelsize": 7,     # tick labels
    "ytick.labelsize": 7,
    "legend.fontsize": 8,     # legends (if used)
    "figure.titlesize": 10
})
# --- Apply IBM Plex Sans globally in all charts ---
plt.rcParams.update({
    "font.family": "IBM Plex Sans",
    "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans", "Arial"],
    "axes.titleweight": "600",
    "axes.labelweight": "500",
    "font.weight": "400"
})


sns.set_style("whitegrid", {
    "axes.facecolor": "#f8fafc",
    "grid.color": "#e6eef8"
})

# ---- Load Data ----
@st.cache_data
def load_data(path="influencer_ranking.csv"):
    df = pd.read_csv(path)
    # ensure expected numeric columns exist
    for c in ["CombinedScore", "EmbeddingNorm", "PageRank", "Betweenness"]:
        if c not in df.columns:
            df[c] = 0.0
    df = df.sort_values("CombinedScore", ascending=False).reset_index(drop=True)
    return df

try:
    df = load_data()
except Exception as e:
    st.error("Could not load 'influencer_ranking.csv'. Ensure file is present next to this script.")
    st.stop()

# ---- Sidebar Controls ----
st.sidebar.title("Dashboard Controls")
top_k = st.sidebar.slider("Number of communities", 5, 50, 20)
keyword = st.sidebar.text_input("Filter by keyword", placeholder="Enter keyword...")
st.sidebar.markdown("---")
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")

# ---- Main Title & Description ----
st.title("Influencer Graph Analytics Dashboard")
st.markdown("""
Comprehensive analysis of influential Reddit communities using **GraphSAGE + GAT** hybrid neural architecture.  
This dashboard blends learned graph embeddings with classic centralities (PageRank, Betweenness) for deeper insights.
""")

# ---- Filter Data ----
if keyword:
    filtered_df = df[df["Node"].str.contains(keyword, case=False, na=False)].head(top_k)
else:
    filtered_df = df.head(top_k)

# ---- Executive Summary Metrics ----
st.markdown("## Executive Summary")
m1, m2, m3 = st.columns(3)
m1.metric("Top Community", filtered_df.iloc[0]["Node"] if len(filtered_df) > 0 else "—")
m2.metric("Maximum Score", f"{filtered_df['CombinedScore'].max():.4f}" if len(filtered_df) > 0 else "0.0000")
m3.metric("Top 10 Average", f"{df['CombinedScore'].head(10).mean():.4f}")

st.divider()

# ---- Tabs ----
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Rankings",
    "Distribution Analysis",
    "Correlation Matrix",
    "Metric Explorer",
    "Network Visualization"
])

# ========== TAB 1: Rankings ==========
with tab1:
    st.header("Community Influence Rankings")
    st.markdown("""
    Rankings of Reddit communities by combined influence score. Use the sidebar to filter and select the number of communities.
    """)
    st.dataframe(
        filtered_df.style.format({
            'CombinedScore': '{:.4f}',
            'EmbeddingNorm': '{:.4f}',
            'PageRank': '{:.6f}',
            'Betweenness': '{:.6f}'
        }),
        use_container_width=True,
        height=480
    )
    csv_data = df.to_csv(index=False)
    st.download_button(
        label="Export Data (CSV)",
        data=csv_data,
        file_name=f"influencer_ranking_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# ========== TAB 2: Distribution ==========
with tab2:
    st.header("Score Distribution Analysis")

    if filtered_df.empty:
        st.warning("No data available for current filters.")
    else:
        # Histogram (Matplotlib)
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.hist(filtered_df["CombinedScore"], bins=25, color='#0f172a', edgecolor='white', alpha=0.92)
        ax.set_xlabel("Combined Influence Score", fontsize=11, fontweight='600', color='#0f172a')
        ax.set_ylabel("Frequency", fontsize=11, fontweight='600', color='#0f172a')
        ax.set_title("Distribution of Combined Influence Scores", fontsize=14, fontweight='700', color='#0f172a', pad=12)
        ax.tick_params(colors='#0f172a')
        ax.grid(alpha=0.25, linestyle='--', linewidth=0.6)
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#f8fafc')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)


        st.markdown("### Top Communities Comparison")

        top_bar = filtered_df.head(top_k)
        fig2, ax2 = plt.subplots()
        ax2.barh(top_bar["Node"], top_bar["CombinedScore"], color='#0f172a', edgecolor='#243045', alpha=0.95)
        ax2.set_xlabel("Combined Influence Score", fontsize=11, fontweight='600', color='#0f172a')
        ax2.set_ylabel("Community", fontsize=11, fontweight='600', color='#0f172a')
        ax2.set_title(f"Top {len(top_bar)} Communities by Combined Score", fontsize=14, fontweight='700', color='#0f172a', pad=12)
        ax2.tick_params(colors='#0f172a')
        ax2.invert_yaxis()
        fig2.patch.set_facecolor('#ffffff')
        ax2.set_facecolor('#f8fafc')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.9)
        st.pyplot(fig2, use_container_width=True)



# ========== TAB 3: Correlation ==========
with tab3:
    st.header("Metric Correlation Analysis")
    st.markdown("Correlation matrix between embedding norms, PageRank, Betweenness and combined score.")

    if filtered_df.empty:
        st.warning("No data available for current filters.")
    else:
        corr = filtered_df[["CombinedScore", "EmbeddingNorm", "PageRank", "Betweenness"]].corr()
        fig3, ax3 = plt.subplots(figsize=(4.5, 2.5))
        sns.heatmap(
            corr, annot=True, cmap="Blues", fmt=".3f", ax=ax3,
            linewidths=0.8, cbar_kws={'shrink': 0.75}, annot_kws={'fontsize': 7, 'fontweight': '500'}
        )
        ax3.set_title("Correlation Matrix", fontsize=11, color='#0f172a', fontweight='700')
        ax3.tick_params(colors='#0f172a')
        fig3.patch.set_facecolor('#ffffff')
        ax3.set_facecolor('#f8fafc')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.9)
        st.pyplot(fig3, use_container_width=True)



# ========== TAB 4: Metric Explorer ==========
with tab4:
    st.header("Metric Relationship Explorer")
    st.markdown("Select any two metrics to analyze their relationship and see correlation trends.")

    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X-Axis Metric", ["EmbeddingNorm", "PageRank", "Betweenness"])
    with col2:
        y_axis = st.selectbox("Y-Axis Metric", ["CombinedScore", "EmbeddingNorm", "PageRank", "Betweenness"], index=0)

    if filtered_df.empty:
        st.warning("No data available for current filters.")
    else:
        # ---- Compute correlation ----
        corr_value = filtered_df[x_axis].corr(filtered_df[y_axis])
        st.markdown(f"**Correlation between {x_axis} and {y_axis}:** `{corr_value:.3f}`")

        # ---- Scatterplot with trendline ----
        fig4, ax4 = plt.subplots(figsize=(4, 2))
        dot_size = max(10, 300 / len(filtered_df))
        sc = ax4.scatter(filtered_df[x_axis], filtered_df[y_axis],
                        c=filtered_df["CombinedScore"], cmap='Blues',
                        s=dot_size, edgecolors='#0f172a', linewidth=0.4, alpha=0.9)
        z = np.polyfit(filtered_df[x_axis], filtered_df[y_axis], 1)
        p = np.poly1d(z)
        ax4.plot(filtered_df[x_axis], p(filtered_df[x_axis]), color="#0f172a", linestyle="--", linewidth=2, alpha=0.8)
        ax4.set_xlabel(x_axis, fontsize=9, fontweight='600', color='#0f172a')
        ax4.set_ylabel(y_axis, fontsize=9, fontweight='600', color='#0f172a')
        ax4.set_title(f"{y_axis} vs {x_axis}", fontsize=11, fontweight='700', color='#0f172a', pad=10)
        ax4.grid(alpha=0.25, linestyle='--')
        ax4.tick_params(colors='#0f172a')
        cbar = plt.colorbar(sc, ax=ax4)
        cbar.set_label('Combined Score', color='#0f172a', fontsize=9, fontweight='600')
        cbar.ax.yaxis.set_tick_params(color='#0f172a')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#0f172a')
        fig4.patch.set_facecolor('#ffffff')
        ax4.set_facecolor('#f8fafc')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.9)
        st.pyplot(fig4, use_container_width=True)


# ========== TAB 5: Network ==========
with tab5:
    st.header("Network Graph Visualization")
    st.markdown("Interactive network showing community relationships. Node size = CombinedScore; color = PageRank.")

    if filtered_df.empty:
        st.warning("No data available for current filters.")
    else:
        # Build network from filtered data
        G = nx.barabasi_albert_graph(max(1, len(filtered_df)), 2, seed=42)

        # Attach node attributes
        for i, n in enumerate(G.nodes()):
            if i < len(filtered_df):
                row = filtered_df.iloc[i]
                G.nodes[n]["label"] = row["Node"]
                G.nodes[n]["score"] = float(row["CombinedScore"])
                G.nodes[n]["pagerank"] = float(row.get("PageRank", 0.0))
            else:
                G.nodes[n]["label"] = f"Node {i}"
                G.nodes[n]["score"] = 0.0
                G.nodes[n]["pagerank"] = 0.0

        pos = nx.spring_layout(G, seed=42, k=0.7)

        edge_x, edge_y = [], []
        for e in G.edges():
            x0, y0 = pos[e[0]]
            x1, y1 = pos[e[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        # Make edges thicker + visible
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.2, color='rgba(15,23,42,0.3)'),  # darker, more visible
            hoverinfo='none',
            mode='lines'
        )

        node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
        for n in G.nodes():
            x, y = pos[n]
            node_x.append(x)
            node_y.append(y)
            meta = G.nodes[n]
            node_text.append(f"<b>{meta['label']}</b><br>Score: {meta['score']:.4f}<br>PageRank: {meta['pagerank']:.6f}")
            node_size.append(12 + meta['score'] * 35)
            node_color.append(meta['pagerank'])

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[G.nodes[n]['label'] if i < 10 else "" for i, n in enumerate(G.nodes())],
            textposition='top center',
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Viridis',
                reversescale=False,
                line=dict(width=1.5, color='#0f172a'),  # border for visibility
                colorbar=dict(title=dict(text="PageRank", side="right"))
            ),
            textfont=dict(size=10, color='#0f172a')
        )

        fig5 = go.Figure(data=[edge_trace, node_trace])
        fig5.update_layout(
            showlegend=False,
            margin=dict(b=20, l=20, r=20, t=30),
            plot_bgcolor='#ffffff',
            paper_bgcolor='#f1f5f9',
            font=dict(color='#0f172a'),
            hoverlabel=dict(bgcolor='#f8fafc', font_color='#0f172a'),
        )
        st.plotly_chart(fig5, use_container_width=True)

# ---- Footer ----
st.divider()
# ---- Global Footer ----
st.markdown("""
<style>
.custom-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    backdrop-filter: blur(8px);
    background: rgba(15, 23, 42, 0.95);
    color: #f1f5f9;
    text-align: center;
    padding: 12px 10px;
    font-size: 0.9rem;
    font-family: 'IBM Plex Sans', sans-serif;
    border-top: 1px solid #334155;
    box-shadow: 0 -2px 8px rgba(0,0,0,0.25);
    z-index: 999999;
    letter-spacing: 0.3px;
    transition: transform 0.5s ease, opacity 0.6s ease;
    animation: fadeInFooter 1.5s ease-out forwards;
    transform: translateY(100%);
    opacity: 0;
}

@keyframes fadeInFooter {
    from {transform: translateY(100%); opacity: 0;}
    to {transform: translateY(0); opacity: 1;}
}

.custom-footer.hidden {
    transform: translateY(100%);
    opacity: 0;
}

/* Ensure sidebar stays above but doesn’t overlap footer */
[data-testid="stSidebar"] {
    z-index: 99998 !important;
}
</style>

<div id="smartFooter" class="custom-footer">
    © 2025 <b>Indu Sree N.</b> | Crafted with curiosity, caffeine, and just enough chaos ☕💻✨ | Powered by <b>Streamlit</b> ⚡
</div>

<script>
// Fade-in is handled by CSS animation. 
// JS adds smart hide/show on scroll.
let lastScrollY = window.scrollY;
const footer = document.getElementById('smartFooter');
let ticking = false;

window.addEventListener('scroll', () => {
    if (!footer) return;
    if (!ticking) {
        window.requestAnimationFrame(() => {
            if (window.scrollY > lastScrollY + 10) {
                footer.classList.add('hidden');
            } else if (window.scrollY < lastScrollY - 10) {
                footer.classList.remove('hidden');
            }
            lastScrollY = window.scrollY;
            ticking = false;
        });
        ticking = true;
    }
});
</script>
""", unsafe_allow_html=True)
