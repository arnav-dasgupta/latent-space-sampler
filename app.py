import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import numpy as np

TRANSPARENT_BG = 'rgba(0,0,0,0)'

# Safety Configuration
MAX_CHARS = 1000

st.set_page_config(page_title="Latent Space Sampler", layout="wide", initial_sidebar_state="collapsed")


st.markdown("""
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .block-container { 
            padding-top: 4rem !important; 
            background-color: #050505; 
        }
        [data-testid="stStatusWidget"] {
            top: 1rem !important;
        }
        [data-testid="column"] { z-index: 10; }
        .stTextArea textarea { background-color: #111; border: 1px solid #333; color: #eee; }
        footer, header {display: none;}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return SentenceTransformer('all-mpnet-base-v2')

model = load_model()

left_col, right_col = st.columns([1, 2.5], gap="large")

with left_col:
    
    st.markdown("""
        <div class="mb-8 border-l-4 border-cyan-500 pl-5 py-2">
            <h1 class="text-3xl font-black text-white tracking-tight uppercase leading-none mb-1">
                Latent Space <span class="text-cyan-500 text-xl block mt-1">Sampler</span>
            </h1>
            <p class="text-zinc-400 text-sm mt-3 leading-relaxed">
                Visualize semantic similarity between text samples in 3D embedding space using sentence transformers.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    text_a = st.text_area("SAMPLE_ALPHA", "Batman is Bruce Wayne", height=90, max_chars=MAX_CHARS)
    text_b = st.text_area("SAMPLE_BETA", "Superman is Clarke Kent", height=90, max_chars=MAX_CHARS)
    
    # Validation logic
    over_limit = len(text_a) >= MAX_CHARS or len(text_b) >= MAX_CHARS
    if over_limit:
        st.error(f"⚠️ Limit: {MAX_CHARS} characters per sample to prevent memory overflow.")
    
    analyze = st.button("RUN VECTOR ANALYSIS")
    
    metric_placeholder = st.empty()

with right_col:
    # Added padding-top to ensure graph toolbar is never obscured
    st.markdown('<div class="mt-8">', unsafe_allow_html=True)
    plot_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# Context anchors
anchors = ["Global finance", "Molecular biology", "Abstract art"]

if analyze or text_a:
    all_texts = [text_a, text_b] + anchors
    embeddings = model.encode(all_texts)
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(embeddings)
    cos_sim = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))

    metric_placeholder.markdown(f"""
        <div class="bg-zinc-900/80 p-4 rounded border border-zinc-800 mt-4 font-mono">
            <p class="text-zinc-500 text-[10px] uppercase font-bold tracking-widest">Similarity</p>
            <p class="text-3xl font-bold text-cyan-400">{(cos_sim*100):.2f}%</p>
        </div>
    """, unsafe_allow_html=True)

    fig = go.Figure()

    for i in range(len(all_texts)):
        is_anchor = i >= 2
        if i == 0:
            lbl, clr = ("ALPHA", "#00f5ff")
        elif i == 1:
            lbl, clr = ("BETA", "#a855f7")
        else:
            lbl, clr = ("ANCHOR", "#3f3f46")
        
        fig.add_trace(go.Scatter3d(
            x=[0, coords[i, 0]], y=[0, coords[i, 1]], z=[0, coords[i, 2]],
            mode='lines',
            line={"color": clr, "width": 7 if not is_anchor else 2},
            hovertext=all_texts[i], hoverinfo='text'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[coords[i, 0]], y=[coords[i, 1]], z=[coords[i, 2]],
            mode='markers+text' if not is_anchor else 'markers',
            marker={"size": 12 if not is_anchor else 5, "color": clr, "opacity": 1 if not is_anchor else 0.4},
            text=[lbl] if not is_anchor else None,
            textposition="top center", textfont={"color": "white"},
            hovertext=all_texts[i], hoverinfo='text'
        ))

    fig.update_layout(
        height=750,
        margin={"l": 0, "r": 0, "b": 0, "t": 60}, # Generous top margin for toolbar
        scene={
            "xaxis": {"showbackground": False, "showgrid": True, "gridcolor": '#222', "title": ""},
            "yaxis": {"showbackground": False, "showgrid": True, "gridcolor": '#222', "title": ""},
            "zaxis": {"showbackground": False, "showgrid": True, "gridcolor": '#222', "title": ""},
            "bgcolor": TRANSPARENT_BG,
            "camera": {"eye": {"x": 1.6, "y": 1.6, "z": 1.6}}
        },
        paper_bgcolor=TRANSPARENT_BG,
        template="plotly_dark",
        showlegend=False,
        modebar={"bgcolor": TRANSPARENT_BG, "color": '#666', "activecolor": '#00f5ff'}
    )

    plot_placeholder.plotly_chart(fig, use_container_width=True)