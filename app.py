import streamlit as st
import torch
import tempfile
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from explainer import (
    load_local_model,
    predict_single_frame,
    predict_and_explain,
    extract_frames_from_video,
    compute_facial_motion_analysis,
    get_motion_verdict
)
from llm_explainer import generate_explanation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def get_model():
    return load_local_model('best_model.pth')

st.set_page_config(page_title="Deepfake Detector", layout="wide")
st.title("🔍 Deepfake Detection")

model = get_model()

uploaded_file = st.file_uploader("Upload image/video", type=['jpg','png','mp4','avi'])

if uploaded_file:
    ext = uploaded_file.name.split('.')[-1].lower()
    is_video = ext in ['mp4','avi']

    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    # ================= VIDEO =================
    if is_video:

        frames, _, duration, _ = extract_frames_from_video(path)
        os.unlink(path)

        if not frames:
            st.error("No frames extracted")
            st.stop()

        results = [predict_single_frame(f, model) for f in frames]
        labels = [r['label'] for r in results]

        flow_results, flow_score = compute_facial_motion_analysis(frames)
        flow_verdict = get_motion_verdict(flow_score)

        total = len(labels)
        fake_count = labels.count('FAKE')

        # ---------- Decision ----------
        model_label = "FAKE" if fake_count >= total/2 else "REAL"
        motion_label = "FAKE" if flow_score >= 50 else "REAL"

        final = "FAKE" if motion_label == "FAKE" else (
            "FAKE" if model_label=="FAKE" and fake_count>0.7*total else "REAL"
        )

        st.subheader(f"Final Verdict: {final}")

        # ================= VISUAL =================
        res = predict_and_explain(frames[0], model)
        c1, c2 = st.columns(2)
        c1.image(res['face_image'], caption="Face")
        c2.image(res['heatmap'], caption="Grad-CAM")

        # ================= GRAPH + SUMMARY =================
        st.subheader("Facial Motion Analysis")

        col1, col2 = st.columns([2,1])

        # ---- Graph ----
        with col1:
            pairs = [r['pair'] for r in flow_results]
            scores = [r['norm_score'] for r in flow_results]
            colors = ['#e74c3c' if s>=50 else '#27ae60' for s in scores]

            fig, ax = plt.subplots()
            ax.bar(range(len(scores)), scores, color=colors)
            ax.axhline(50, linestyle='--')
            ax.set_xticks(range(len(pairs)))
            ax.set_xticklabels(pairs, rotation=45, fontsize=8)
            ax.set_ylabel("Irregularity (%)")
            st.pyplot(fig)

        # ---- Summary ----
        with col2:
            color = "#c0392b" if flow_score>=60 else "#e67e22" if flow_score>=35 else "#1e8449"

            st.markdown(f"""
            **Score:** <span style='color:{color};font-size:24px'>{flow_score}%</span>  
            **Verdict:** {flow_verdict}
            """, unsafe_allow_html=True)

        # ================= EXPLANATION =================
        st.subheader("Explanation")

        exp = generate_explanation(
            final, flow_score, total, fake_count, duration,
            flow_score=flow_score, flow_verdict=flow_verdict
        )

        st.write(exp)

    # ================= IMAGE =================
    else:
        img = Image.open(path).convert('RGB')
        os.unlink(path)

        res = predict_and_explain(img, model)

        st.subheader(f"Final Verdict: {res['label']}")

        c1, c2 = st.columns(2)
        c1.image(res['face_image'], caption="Face")
        c2.image(res['heatmap'], caption="Grad-CAM")

        st.write("Static image — motion analysis not applicable.")