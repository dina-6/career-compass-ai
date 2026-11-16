import streamlit as st
from pypdf import PdfReader
from src.analyzer import load_model, recommend_roles, skill_gap

st.set_page_config(page_title="CareerCompass AI", page_icon="🧭", layout="wide")
st.title("🧭 CareerCompass AI")
st.subheader("Resume-based career role and skill-gap recommender")
st.write("Paste your resume text or upload a PDF. The model suggests related career domains and highlights skills to learn.")

uploaded = st.file_uploader("Upload resume (PDF)", type=["pdf"])
resume_text = st.text_area("Or paste resume / skills here", height=220,
    placeholder="Example: Python, SQL, statistics, pandas, machine learning...")
if uploaded:
    try:
        reader = PdfReader(uploaded)
        pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if pdf_text.strip():
            resume_text = pdf_text
            st.success("PDF text extracted. You can review or edit the text in the box above.")
        else:
            st.warning("No selectable text found. Try a text-based PDF or paste your resume content.")
    except Exception as exc:
        st.error(f"Could not read PDF: {exc}")

if st.button("Analyse my career fit", type="primary", disabled=not resume_text.strip()):
    try:
        model = load_model()
        recommendations = recommend_roles(resume_text, model)
        st.markdown("### Recommended career domains")
        for item in recommendations:
            with st.container(border=True):
                st.write(f"**{item['role']}**")
                st.progress(item["score"], text=f"Model score: {item['score']:.1%}")
                gap = skill_gap(resume_text, item["role"])
                left, right = st.columns(2)
                with left:
                    st.markdown("**Skills found**")
                    st.write(", ".join(gap["matched"]) if gap["matched"] else "No listed skills detected yet")
                with right:
                    st.markdown("**Skills to consider learning**")
                    st.write(", ".join(gap["missing"]) if gap["missing"] else "All listed skills detected")
        st.info("Use these suggestions as a starting point. A text classifier cannot assess your full ability, interests, experience, or job-market fit.")
    except FileNotFoundError as exc:
        st.warning(str(exc))
        st.code("python train.py")

st.divider()
st.caption("Academic portfolio prototype • TF-IDF • Logistic Regression • Streamlit")
