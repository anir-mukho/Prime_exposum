"""
Exponential Sum over Prime Numbers — Streamlit Visualizer
=========================================================
Evaluates  S(α) = Σ_{p ≤ N} e^{i·p·α}  for all primes p ≤ N
and plots the real part, imaginary part, or absolute magnitude
as a function of α ∈ [0, 2π].
"""

import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Prime Exponential Sum Visualizer",
    page_icon="🔢",
    layout="wide",
)

# ──────────────────────────────────────────────
# Custom CSS for a premium dark aesthetic
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ───────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ──────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #58a6ff;
    }

    /* ── Header banner ────────────────────── */
    .hero-banner {
        background: linear-gradient(135deg, #0d1117 0%, #1a1e2e 50%, #1c1236 100%);
        border: 1px solid rgba(88, 166, 255, .15);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .hero-banner h1 {
        background: linear-gradient(90deg, #79c0ff, #d2a8ff, #ff7b72);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 .4rem;
    }
    .hero-banner p {
        color: #8b949e;
        font-size: 1rem;
        margin: 0;
    }

    /* ── Stat cards ───────────────────────── */
    .stat-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.2rem;
    }
    .stat-card {
        flex: 1;
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        border: 1px solid rgba(88, 166, 255, .12);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .stat-card .label {
        color: #8b949e;
        font-size: .75rem;
        text-transform: uppercase;
        letter-spacing: .06em;
    }
    .stat-card .value {
        color: #58a6ff;
        font-size: 1.6rem;
        font-weight: 700;
    }

    /* ── Misc polish ──────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: .55rem 2rem !important;
        font-weight: 600 !important;
        transition: transform .15s ease, box-shadow .15s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(35, 134, 54, .45) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Core math utilities (cached)
# ──────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def sieve_of_eratosthenes(n: int) -> np.ndarray:
    """Return an array of all primes ≤ *n* via the Sieve of Eratosthenes."""
    if n < 2:
        return np.array([], dtype=np.int64)
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[:2] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            is_prime[i * i :: i] = False
    return np.nonzero(is_prime)[0].astype(np.int64)


@st.cache_data(show_spinner=False)
def exponential_sum(primes: tuple, alpha: np.ndarray) -> np.ndarray:
    """Compute S(α) = Σ_{p in primes} exp(i·p·α) for each α value."""
    p = np.array(primes, dtype=np.float64)
    # Outer product → shape (len(alpha), len(primes))
    phases = np.outer(alpha, p)
    return np.sum(np.exp(1j * phases), axis=1)


def compute_sum_for_n(n: int, alpha: np.ndarray):
    """Helper: sieve + sum in one call (uses caching under the hood)."""
    primes = sieve_of_eratosthenes(n)
    return exponential_sum(tuple(primes), alpha), primes


# ──────────────────────────────────────────────
# Plotly chart builder
# ──────────────────────────────────────────────

COMPONENT_MAP = {
    "Real Part  Re[S(α)]": ("real", "Re[S(α)]", "#79c0ff"),
    "Imaginary Part  Im[S(α)]": ("imag", "Im[S(α)]", "#d2a8ff"),
    "Absolute Magnitude  |S(α)|": ("abs", "|S(α)|", "#ff7b72"),
}


def build_figure(alpha, s_alpha, component_key, n_value):
    kind, y_label, color = COMPONENT_MAP[component_key]
    if kind == "real":
        y = s_alpha.real
    elif kind == "imag":
        y = s_alpha.imag
    else:
        y = np.abs(s_alpha)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=alpha,
            y=y,
            mode="lines",
            line=dict(color=color, width=2.2),
            fill="tozeroy",
            fillcolor=color.replace(")", ", 0.08)").replace("rgb", "rgba")
            if color.startswith("rgb")
            else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
            hovertemplate="α = %{x:.4f}<br>" + y_label + " = %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        title=dict(
            text=f"S(α) for primes p ≤ {n_value}",
            font=dict(size=20, color="#c9d1d9"),
            x=0.5,
        ),
        xaxis=dict(
            title="α  (0 → 2π)",
            gridcolor="rgba(88,166,255,0.08)",
            zerolinecolor="rgba(88,166,255,0.15)",
            tickvals=[0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
            ticktext=["0", "π/2", "π", "3π/2", "2π"],
        ),
        yaxis=dict(
            title=y_label,
            gridcolor="rgba(88,166,255,0.08)",
            zerolinecolor="rgba(88,166,255,0.15)",
        ),
        margin=dict(l=60, r=30, t=60, b=50),
        height=520,
        font=dict(family="Inter, sans-serif", color="#c9d1d9"),
        hoverlabel=dict(
            bgcolor="#161b22",
            bordercolor="#30363d",
            font_color="#c9d1d9",
        ),
    )
    return fig


# ──────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")

    mode = st.selectbox(
        "Mode",
        ["Single N Plot", "Animate Range of N"],
        help="Choose between a static plot or an animated sweep.",
    )

    component = st.selectbox(
        "Component to plot",
        list(COMPONENT_MAP.keys()),
    )

    num_alpha = st.slider(
        "α resolution (steps)",
        min_value=500,
        max_value=2000,
        value=800,
        step=100,
        help="Number of sample points in [0, 2π].",
    )

    st.markdown("---")

    if mode == "Single N Plot":
        n_value = st.number_input(
            "N  (max prime bound)",
            min_value=2,
            max_value=2000,
            value=100,
            step=10,
        )
    else:
        n_range = st.slider(
            "N range  (Start → End)",
            min_value=2,
            max_value=2000,
            value=(10, 500),
            step=10,
        )
        anim_speed = st.slider(
            "Animation speed (seconds per frame)",
            min_value=0.05,
            max_value=1.0,
            value=0.15,
            step=0.05,
        )
        anim_step = st.slider(
            "N increment per frame",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
        )
        start_anim = st.button("▶  Start Animation", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<div style='color:#8b949e;font-size:.78rem'>"
        "Built with Streamlit + Plotly<br>"
        "© 2026 Prime Exp-Sum Viz"
        "</div>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# Alpha array (shared)
# ──────────────────────────────────────────────
alpha = np.linspace(0, 2 * np.pi, num_alpha, endpoint=True)

# ──────────────────────────────────────────────
# Hero banner
# ──────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <h1>🔢 Prime Exponential Sum Visualizer</h1>
        <p>Explore <em>S(α) = Σ<sub>p≤N</sub> e<sup>ipα</sup></em> — the interference pattern of primes</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Main content
# ──────────────────────────────────────────────

if mode == "Single N Plot":
    s_alpha, primes = compute_sum_for_n(int(n_value), alpha)

    # Stat cards
    st.markdown(
        f"""
        <div class="stat-row">
            <div class="stat-card">
                <div class="label">N (bound)</div>
                <div class="value">{int(n_value)}</div>
            </div>
            <div class="stat-card">
                <div class="label">Primes found</div>
                <div class="value">{len(primes)}</div>
            </div>
            <div class="stat-card">
                <div class="label">Largest prime</div>
                <div class="value">{primes[-1] if len(primes) else '—'}</div>
            </div>
            <div class="stat-card">
                <div class="label">Max |S(α)|</div>
                <div class="value">{np.max(np.abs(s_alpha)):.2f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = build_figure(alpha, s_alpha, component, int(n_value))
    st.plotly_chart(fig, use_container_width=True)

else:
    # ── Animation mode ────────────────────────
    stat_placeholder = st.empty()
    chart_placeholder = st.empty()
    progress_bar = st.empty()

    if start_anim:
        n_start, n_end = n_range
        n_values = list(range(n_start, n_end + 1, anim_step))
        total = len(n_values)

        for idx, n_val in enumerate(n_values):
            s_alpha, primes = compute_sum_for_n(n_val, alpha)

            stat_placeholder.markdown(
                f"""
                <div class="stat-row">
                    <div class="stat-card">
                        <div class="label">Current N</div>
                        <div class="value">{n_val}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Primes ≤ N</div>
                        <div class="value">{len(primes)}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Largest prime</div>
                        <div class="value">{primes[-1] if len(primes) else '—'}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Max |S(α)|</div>
                        <div class="value">{np.max(np.abs(s_alpha)):.2f}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            fig = build_figure(alpha, s_alpha, component, n_val)
            chart_placeholder.plotly_chart(fig, use_container_width=True)

            progress_bar.progress(
                (idx + 1) / total,
                text=f"Frame {idx + 1}/{total}  —  N = {n_val}",
            )
            time.sleep(anim_speed)

        progress_bar.progress(1.0, text="✅ Animation complete!")
    else:
        st.info("Configure the range and speed in the sidebar, then press **▶ Start Animation**.")
