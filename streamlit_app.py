"""
streamlit_app.py — Interactive UI for testing the AI Fashion Stylist pipeline.

HOW TO RUN:
  streamlit run streamlit_app.py

WHAT IT DOES:
  Provides a beautiful UI to test all three pipeline modes:
    1. Full Pipeline   — Upload photo → get full recommendation
    2. Vision Only     — Upload photo → see extracted profile
    3. Manual Profile  — Enter profile manually → get recommendation

IMPORTANT: The FastAPI server does NOT need to be running.
  This app calls the AI pipeline modules DIRECTLY (Python imports).
  This makes testing faster and simpler.
"""

import sys
import json
import io
import time
import logging
from pathlib import Path

import streamlit as st
from PIL import Image

# ── Add project root to path ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Page config (MUST be first Streamlit call) ────────────────────────
st.set_page_config(
    page_title="AI Fashion Stylist — Test UI",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Cards */
.glass-card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}

/* Section headers */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #c084fc;
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Profile badge */
.profile-badge {
    display: inline-block;
    background: linear-gradient(135deg, #7c3aed, #db2777);
    color: white;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 4px;
}

/* Confidence bar container */
.conf-label {
    font-size: 0.8rem;
    color: #a78bfa;
    margin-bottom: 2px;
}

/* Color chip */
.color-chip {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 500;
    margin: 3px;
    border: 1px solid rgba(255,255,255,0.2);
    color: white;
    background: rgba(124, 58, 237, 0.4);
}

.color-chip-avoid {
    background: rgba(220, 38, 38, 0.3);
    border-color: rgba(220, 38, 38, 0.4);
}

/* Outfit card */
.outfit-card {
    background: rgba(255,255,255,0.05);
    border-left: 3px solid #7c3aed;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin: 10px 0;
}

/* Metric box */
.metric-box {
    background: rgba(124, 58, 237, 0.15);
    border: 1px solid rgba(124, 58, 237, 0.3);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}

/* Success banner */
.success-banner {
    background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(5,150,105,0.2));
    border: 1px solid rgba(16,185,129,0.4);
    border-radius: 12px;
    padding: 16px 20px;
    color: #6ee7b7;
    font-weight: 500;
}

/* Warning box */
.warn-box {
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-radius: 10px;
    padding: 14px 18px;
    color: #fcd34d;
    font-size: 0.9rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.4) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #db2777) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 2rem !important;
    transition: opacity 0.2s;
}
.stButton > button:hover {
    opacity: 0.88 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: rgba(255,255,255,0.5);
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    color: #c084fc !important;
}

/* Hide streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# HELPER: Safe import of AI modules (with friendly error if .env missing)
# ════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_ai_modules():
    """Load AI modules once, cache them. Returns (success, error_msg)."""
    try:
        from config import settings
        from ai_engine.vision_analyzer import analyze_photo
        from ai_engine.recommender import generate_recommendation, run_full_pipeline
        from ai_engine.rag_pipeline import retrieve_fashion_rules
        return True, None, {
            "settings": settings,
            "analyze_photo": analyze_photo,
            "generate_recommendation": generate_recommendation,
            "run_full_pipeline": run_full_pipeline,
            "retrieve_fashion_rules": retrieve_fashion_rules,
        }
    except Exception as e:
        return False, str(e), {}


def check_chroma_status():
    """Check if ChromaDB knowledge base is built."""
    try:
        from config import settings
        import chromadb
        client = chromadb.PersistentClient(path=settings.chroma_db_path)
        collection = client.get_collection(name=settings.chroma_collection_name)
        count = collection.count()
        return True, count
    except Exception as e:
        return False, 0


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 👗 Fashion Stylist")
    st.markdown("**AI Test Interface**")
    st.divider()

    # System status
    st.markdown("### 🔧 System Status")

    ok, err, modules = load_ai_modules()
    if ok:
        st.success("✅ AI Modules Loaded")
        from config import settings
        st.caption(f"Vision: `{settings.gemini_vision_model}`")
        st.caption(f"LLM: `{settings.gemini_llm_model}`")
        st.caption(f"Embed: `models/text-embedding-004`")
    else:
        st.error("❌ AI Modules Failed")
        st.caption(f"Error: {err}")

    chroma_ok, doc_count = check_chroma_status()
    if chroma_ok:
        st.success(f"✅ ChromaDB: {doc_count} docs")
    else:
        st.warning("⚠️ ChromaDB not built")
        st.caption("Run: `python knowledge_base/builder.py`")

    st.divider()

    # Settings
    st.markdown("### ⚙️ Settings")
    prompt_version = st.selectbox(
        "Vision Prompt Version",
        ["v2", "v1", "v3"],
        help="v2=production, v1=baseline, v3=chain-of-thought"
    )
    n_rag = st.slider(
        "RAG Results (k)",
        min_value=1, max_value=10, value=5,
        help="How many fashion rules to retrieve from ChromaDB"
    )

    st.divider()
    st.markdown("### 📖 Info")
    st.caption(
        "This UI directly calls the Python modules — "
        "FastAPI server does not need to run."
    )
    st.caption("**Project:** AI Fashion Stylist | B.Tech FYP")


# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem 0;">
    <h1 style="font-size:2.8rem; font-weight:700;
               background: linear-gradient(135deg, #c084fc, #f472b6, #fb923c);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        👗 AI Fashion Stylist
    </h1>
    <p style="color: rgba(255,255,255,0.5); font-size: 1rem; margin-top: -8px;">
        Gemini Vision · ChromaDB RAG · Personalized Recommendations
    </p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# BLOCKER: Show setup instructions if modules not loaded
# ════════════════════════════════════════════════════════════════

if not ok:
    st.markdown("""
    <div class="warn-box">
    ⚠️ <b>Setup Required</b> — AI modules could not be loaded.<br>
    Make sure your <code>.env</code> file exists with a valid 
    <code>GOOGLE_API_KEY</code>.
    </div>
    """, unsafe_allow_html=True)
    st.code("""
# Steps to fix:
# 1. Copy the template
copy .env.example .env

# 2. Open .env and fill in your API key from aistudio.google.com

# 3. Build ChromaDB (one-time)
python knowledge_base/builder.py

# 4. Restart this Streamlit app
streamlit run streamlit_app.py
    """, language="bash")
    st.stop()


# ════════════════════════════════════════════════════════════════
# MAIN TABS
# ════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Full Pipeline",
    "🔍 Vision Only",
    "✏️ Manual Profile",
    "📊 RAG Inspector",
])


# ──────────────────────────────────────────────────────────────
# TAB 1 — FULL PIPELINE (Photo → Vision → RAG → Recommendation)
# ──────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Upload a photo — get a complete style recommendation")
    st.caption(
        "The full pipeline runs: **Gemini Vision** → **ChromaDB RAG** → **Gemini LLM**"
    )

    col_upload, col_preview = st.columns([1, 1], gap="large")

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload a photo (face + body visible)",
            type=["jpg", "jpeg", "png", "webp"],
            key="full_pipeline_upload",
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            with col_preview:
                st.image(image, caption="Uploaded Photo", use_container_width=True)

        if uploaded_file and not chroma_ok:
            st.markdown("""
            <div class="warn-box">
            ⚠️ ChromaDB knowledge base not built yet!<br>
            Run: <code>python knowledge_base/builder.py</code><br>
            The vision step will still work, but RAG will fail.
            </div>
            """, unsafe_allow_html=True)

        run_btn = st.button(
            "✨ Analyze & Recommend",
            disabled=not uploaded_file,
            use_container_width=True,
            key="run_full"
        )

    if run_btn and uploaded_file:
        image_bytes = uploaded_file.getvalue()
        run_full_pipeline = modules["run_full_pipeline"]

        with st.spinner("🔮 Running full pipeline... (may take 10-20 seconds)"):
            progress = st.progress(0, text="Step 1/3: Gemini Vision analyzing photo...")
            try:
                time.sleep(0.3)
                progress.progress(15, text="Step 1/3: Gemini Vision analyzing photo...")

                result = run_full_pipeline(
                    image_source=image_bytes,
                    n_rag_results=n_rag,
                    vision_prompt_version=prompt_version,
                )

                progress.progress(60, text="Step 2/3: ChromaDB retrieving fashion rules...")
                time.sleep(0.3)
                progress.progress(90, text="Step 3/3: Gemini LLM generating recommendation...")
                time.sleep(0.3)
                progress.progress(100, text="✅ Complete!")
                time.sleep(0.4)
                progress.empty()

                st.session_state["full_result"] = result
                st.markdown("""
                <div class="success-banner">
                ✅ Pipeline complete! Scroll down to see your personalized recommendation.
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                progress.empty()
                st.error(f"❌ Pipeline failed: {e}")
                st.exception(e)

    # Display results
    if "full_result" in st.session_state:
        result = st.session_state["full_result"]
        profile = result.get("user_profile", {})
        rec = result.get("recommendation", {})
        rag_meta = result.get("rag_metadata", {})

        st.divider()

        # ── PROFILE SECTION ─────────────────────────────────
        st.markdown("#### 🧬 Detected Profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-box">
                <div style="font-size:2rem">😊</div>
                <div style="color:#a78bfa;font-size:0.75rem;margin:4px 0">FACE SHAPE</div>
                <div style="font-size:1.3rem;font-weight:700;color:white">
                    {profile.get("face_shape","—")}
                </div>
                <div style="color:#6b7280;font-size:0.75rem">
                    {int((profile.get("face_shape_confidence") or 0)*100)}% confidence
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-box">
                <div style="font-size:2rem">🎨</div>
                <div style="color:#a78bfa;font-size:0.75rem;margin:4px 0">SKIN TONE</div>
                <div style="font-size:1.3rem;font-weight:700;color:white">
                    {profile.get("skin_tone","—")}
                </div>
                <div style="color:#6b7280;font-size:0.75rem">
                    {profile.get("skin_undertone","—")} undertone
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-box">
                <div style="font-size:2rem">👤</div>
                <div style="color:#a78bfa;font-size:0.75rem;margin:4px 0">BODY TYPE</div>
                <div style="font-size:1.3rem;font-weight:700;color:white">
                    {profile.get("body_type","—")}
                </div>
                <div style="color:#6b7280;font-size:0.75rem">
                    {int((profile.get("body_type_confidence") or 0)*100)}% confidence
                </div>
            </div>
            """, unsafe_allow_html=True)

        if profile.get("notes"):
            st.caption(f"📝 Vision notes: {profile['notes']}")

        st.divider()

        # ── RECOMMENDATION SECTION ───────────────────────────
        st.markdown("#### 🎯 Personalized Recommendation")

        rc1, rc2 = st.columns(2, gap="large")

        with rc1:
            # Colors
            color_data = rec.get("color_palette", {})
            st.markdown('<div class="section-title">🎨 Color Palette</div>', unsafe_allow_html=True)
            best_colors = color_data.get("best_colors", [])
            if best_colors:
                chips = "".join(
                    f'<span class="color-chip">{c}</span>' for c in best_colors
                )
                st.markdown(chips, unsafe_allow_html=True)
            st.caption(color_data.get("color_explanation", ""))

            avoid_colors = color_data.get("colors_to_avoid", [])
            if avoid_colors:
                st.markdown("**Avoid:**")
                avoid_chips = "".join(
                    f'<span class="color-chip color-chip-avoid">{c}</span>'
                    for c in avoid_colors
                )
                st.markdown(avoid_chips, unsafe_allow_html=True)

            st.divider()

            # Fabrics
            fabrics = rec.get("fabrics", [])
            st.markdown('<div class="section-title">🧵 Fabrics</div>', unsafe_allow_html=True)
            if fabrics:
                fabric_chips = "".join(
                    f'<span class="color-chip">{f}</span>' for f in fabrics
                )
                st.markdown(fabric_chips, unsafe_allow_html=True)

        with rc2:
            # Styles
            style_data = rec.get("clothing_styles", {})
            st.markdown('<div class="section-title">✂️ Clothing Styles</div>', unsafe_allow_html=True)
            for s in style_data.get("recommended", []):
                st.markdown(f"✅ {s}")
            if style_data.get("avoid"):
                with st.expander("Styles to avoid"):
                    for s in style_data.get("avoid", []):
                        st.markdown(f"🚫 {s}")
            st.caption(style_data.get("style_explanation", ""))

            st.divider()

            # Patterns
            pattern_data = rec.get("patterns", {})
            st.markdown('<div class="section-title">🔲 Patterns</div>', unsafe_allow_html=True)
            for p in pattern_data.get("recommended", []):
                st.markdown(f"✅ {p}")
            if pattern_data.get("avoid"):
                with st.expander("Patterns to avoid"):
                    for p in pattern_data.get("avoid", []):
                        st.markdown(f"🚫 {p}")

        # Outfit ideas
        st.divider()
        st.markdown("#### 👗 Outfit Ideas")
        outfit_ideas = rec.get("outfit_ideas", [])
        if outfit_ideas:
            cols = st.columns(min(len(outfit_ideas), 3))
            for i, idea in enumerate(outfit_ideas[:3]):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="outfit-card">
                        <div style="color:#c084fc;font-size:0.75rem;
                                    font-weight:600;text-transform:uppercase">
                            {idea.get("occasion","Outfit")}
                        </div>
                        <div style="color:white;font-weight:600;margin:6px 0">
                            {idea.get("outfit","—")}
                        </div>
                        <div style="color:#9ca3af;font-size:0.82rem">
                            {idea.get("why_it_works","—")}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Stylist note
        stylist_note = rec.get("stylist_note", "")
        if stylist_note:
            st.divider()
            st.markdown(f"""
            <div class="glass-card" style="border-color: rgba(192,132,252,0.3)">
                <div style="font-size:1.5rem">💌</div>
                <div style="color:#e9d5ff;font-style:italic;margin-top:8px">
                    "{stylist_note}"
                </div>
                <div style="color:#7c3aed;font-size:0.8rem;margin-top:8px">
                    — Your AI Stylist
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── RAG METADATA ─────────────────────────────────────
        with st.expander("🔬 RAG Retrieval Details (for debugging/viva)"):
            st.caption(f"**Query sent to ChromaDB:** {rag_meta.get('query_used','—')}")
            st.caption(f"**Rules retrieved:** {rag_meta.get('n_rules_retrieved','—')}")
            st.caption(f"**Rule IDs used:** {', '.join(rag_meta.get('rule_ids_used',[]))}")
            dists = rag_meta.get("retrieval_distances", [])
            if dists:
                import plotly.graph_objects as go
                fig = go.Figure(go.Bar(
                    x=rag_meta.get("rule_ids_used", []),
                    y=dists,
                    marker_color="rgba(124,58,237,0.7)",
                    marker_line_color="rgba(192,132,252,0.8)",
                    marker_line_width=1.5,
                ))
                fig.update_layout(
                    title="Retrieval Distances (lower = more relevant)",
                    xaxis_title="Document ID",
                    yaxis_title="Cosine Distance",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

        # Raw JSON
        with st.expander("📋 Raw JSON Response"):
            st.json(result)


# ──────────────────────────────────────────────────────────────
# TAB 2 — VISION ONLY
# ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Test Gemini Vision Analysis Only")
    st.caption(
        "Upload a photo and see what Gemini extracts — no RAG, no recommendation. "
        "Good for testing prompt versions."
    )

    v_col1, v_col2 = st.columns([1, 1], gap="large")
    with v_col1:
        vision_file = st.file_uploader(
            "Upload photo",
            type=["jpg", "jpeg", "png", "webp"],
            key="vision_only_upload",
        )
        compare_prompts = st.checkbox(
            "Compare all 3 prompt versions",
            help="Runs v1, v2, v3 on the same photo and shows side by side"
        )
        vision_btn = st.button(
            "🔍 Analyze Photo",
            disabled=not vision_file,
            use_container_width=True,
            key="run_vision"
        )

    with v_col2:
        if vision_file:
            st.image(Image.open(vision_file), caption="Preview", use_container_width=True)

    if vision_btn and vision_file:
        analyze_photo = modules["analyze_photo"]
        image_bytes = vision_file.getvalue()

        if compare_prompts:
            st.markdown("#### 🔬 Prompt Version Comparison")
            cols = st.columns(3)
            for col, version in zip(cols, ["v1", "v2", "v3"]):
                with col:
                    st.markdown(f"**Prompt {version.upper()}**")
                    with st.spinner(f"Running {version}..."):
                        try:
                            t0 = time.time()
                            profile = analyze_photo(image_bytes, prompt_version=version)
                            elapsed = round(time.time() - t0, 2)
                            st.success(f"✅ {elapsed}s")
                            st.metric("Face Shape", profile.get("face_shape", "—"))
                            st.metric("Skin Tone", profile.get("skin_tone", "—"))
                            st.metric("Body Type", profile.get("body_type", "—"))
                            st.metric(
                                "Avg Confidence",
                                f"{int(((profile.get('face_shape_confidence') or 0) + (profile.get('skin_tone_confidence') or 0) + (profile.get('body_type_confidence') or 0)) / 3 * 100)}%"
                            )
                            with st.expander("Full JSON"):
                                st.json(profile)
                        except Exception as e:
                            st.error(f"❌ {e}")
        else:
            with st.spinner(f"🔍 Analyzing with prompt {prompt_version}..."):
                try:
                    t0 = time.time()
                    profile = analyze_photo(image_bytes, prompt_version=prompt_version)
                    elapsed = round(time.time() - t0, 2)

                    st.markdown(f"#### ✅ Analysis Result `({elapsed}s)`")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Face Shape", profile.get("face_shape", "—"),
                              f"{int((profile.get('face_shape_confidence') or 0)*100)}%")
                    m2.metric("Skin Tone", profile.get("skin_tone", "—"),
                              f"{int((profile.get('skin_tone_confidence') or 0)*100)}%")
                    m3.metric("Undertone", profile.get("skin_undertone", "—"))
                    m4.metric("Body Type", profile.get("body_type", "—"),
                              f"{int((profile.get('body_type_confidence') or 0)*100)}%")

                    if profile.get("notes"):
                        st.info(f"📝 Notes: {profile['notes']}")

                    st.json(profile)

                except Exception as e:
                    st.error(f"❌ Vision analysis failed: {e}")
                    st.exception(e)


# ──────────────────────────────────────────────────────────────
# TAB 3 — MANUAL PROFILE INPUT
# ──────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Enter Profile Manually → Get Recommendation")
    st.caption(
        "Bypass vision analysis — directly enter face shape, skin tone, body type. "
        "Useful for testing RAG + LLM without a real photo."
    )

    with st.form("manual_profile_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            face_shape = st.selectbox(
                "Face Shape",
                ["Oval", "Round", "Square", "Heart", "Diamond", "Rectangle", "Triangle"]
            )
        with c2:
            skin_tone = st.selectbox(
                "Skin Tone",
                ["Fair", "Light", "Medium", "Olive", "Tan", "Deep"]
            )
            skin_undertone = st.selectbox(
                "Undertone",
                ["Warm", "Cool", "Neutral"]
            )
        with c3:
            body_type = st.selectbox(
                "Body Type",
                ["Hourglass", "Pear", "Apple", "Rectangle", "Inverted Triangle"]
            )

        manual_btn = st.form_submit_button(
            "✨ Get Recommendation",
            use_container_width=True
        )

    if manual_btn:
        if not chroma_ok:
            st.error("❌ ChromaDB not built. Run: `python knowledge_base/builder.py`")
        else:
            generate_recommendation = modules["generate_recommendation"]
            user_profile = {
                "face_shape": face_shape,
                "skin_tone": skin_tone,
                "skin_undertone": skin_undertone,
                "body_type": body_type,
            }
            with st.spinner("🔮 Generating recommendation..."):
                try:
                    result = generate_recommendation(
                        user_profile=user_profile,
                        n_rag_results=n_rag,
                    )
                    st.session_state["manual_result"] = result
                    st.success("✅ Recommendation generated!")
                except Exception as e:
                    st.error(f"❌ Failed: {e}")
                    st.exception(e)

    if "manual_result" in st.session_state:
        result = st.session_state["manual_result"]
        rec = result.get("recommendation", {})

        st.divider()
        st.markdown("#### 🎯 Recommendation")

        col_a, col_b = st.columns(2)
        with col_a:
            colors = rec.get("color_palette", {})
            st.markdown("**🎨 Best Colors**")
            chips = "".join(
                f'<span class="color-chip">{c}</span>'
                for c in colors.get("best_colors", [])
            )
            st.markdown(chips or "_None_", unsafe_allow_html=True)
            st.caption(colors.get("color_explanation", ""))

            st.markdown("**✂️ Recommended Styles**")
            for s in rec.get("clothing_styles", {}).get("recommended", []):
                st.markdown(f"• {s}")

        with col_b:
            st.markdown("**🔲 Patterns**")
            for p in rec.get("patterns", {}).get("recommended", []):
                st.markdown(f"• {p}")

            st.markdown("**🧵 Fabrics**")
            fabric_chips = "".join(
                f'<span class="color-chip">{f}</span>'
                for f in rec.get("fabrics", [])
            )
            st.markdown(fabric_chips or "_None_", unsafe_allow_html=True)

        st.divider()
        st.markdown(f"💌 *\"{rec.get('stylist_note', '')}\"*")

        with st.expander("📋 Full JSON + RAG Details"):
            st.json(result)


# ──────────────────────────────────────────────────────────────
# TAB 4 — RAG INSPECTOR
# ──────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 🔬 RAG Retrieval Inspector")
    st.caption(
        "Test ChromaDB retrieval directly. "
        "See exactly which fashion rules are retrieved for any profile "
        "and their similarity scores."
    )

    if not chroma_ok:
        st.warning("⚠️ ChromaDB not built. Run: `python knowledge_base/builder.py`")
    else:
        ri_c1, ri_c2, ri_c3, ri_c4 = st.columns(4)
        with ri_c1:
            ri_face = st.selectbox("Face Shape", ["Oval","Round","Square","Heart","Diamond","Rectangle","Triangle"], key="ri_face")
        with ri_c2:
            ri_skin = st.selectbox("Skin Tone", ["Fair","Light","Medium","Olive","Tan","Deep"], key="ri_skin")
            ri_undertone = st.selectbox("Undertone", ["Warm","Cool","Neutral"], key="ri_ut")
        with ri_c3:
            ri_body = st.selectbox("Body Type", ["Hourglass","Pear","Apple","Rectangle","Inverted Triangle"], key="ri_body")
        with ri_c4:
            ri_k = st.slider("k (results)", 1, 10, 5, key="ri_k")

        if st.button("🔍 Run Retrieval", key="run_retrieval", use_container_width=True):
            retrieve = modules["retrieve_fashion_rules"]
            with st.spinner("Querying ChromaDB..."):
                try:
                    results = retrieve(
                        face_shape=ri_face,
                        skin_tone=ri_skin,
                        skin_undertone=ri_undertone,
                        body_type=ri_body,
                        n_results=ri_k,
                    )
                    st.session_state["rag_results"] = results
                except Exception as e:
                    st.error(f"❌ Retrieval failed: {e}")

        if "rag_results" in st.session_state:
            results = st.session_state["rag_results"]

            st.markdown(f"""
            <div class="glass-card">
                <b>Query sent to ChromaDB:</b><br>
                <span style="color:#c084fc">{results['query_used']}</span>
            </div>
            """, unsafe_allow_html=True)

            # Distance chart
            import plotly.graph_objects as go
            fig = go.Figure(go.Bar(
                x=results["ids"],
                y=results["distances"],
                marker_color=[
                    f"rgba(124,58,237,{max(0.3, 1 - d)})"
                    for d in results["distances"]
                ],
                marker_line_color="rgba(192,132,252,0.8)",
                marker_line_width=1.5,
                text=[f"{d:.3f}" for d in results["distances"]],
                textposition="outside",
                textfont_color="white",
            ))
            fig.update_layout(
                title="Retrieval Distances (lower = more similar to query)",
                xaxis_title="Document ID",
                yaxis_title="Cosine Distance",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                height=320,
                yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Document cards
            st.markdown(f"#### Retrieved Documents ({results['n_retrieved']})")
            for i, (doc_id, dist, doc, meta) in enumerate(zip(
                results["ids"],
                results["distances"],
                results["documents"],
                results["metadatas"],
            ), 1):
                with st.expander(f"Rank {i}: `{doc_id}` | distance: {dist:.4f} | category: {meta.get('category','?')}"):
                    st.markdown(doc)
                    st.json(meta)
