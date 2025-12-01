# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import keras
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import time

# Compatibility shim:
# Some models saved with older / standalone Keras include keys like
# 'batch_shape' in InputLayer config or serialize dtype policies as
# 'DTypePolicy'. Newer tf.keras expects 'batch_input_shape' and uses
# tf.keras.mixed_precision.Policy for dtype policy. The shim remaps these
# during deserialization so load_model can succeed in many common mismatch cases.
try:
    # prefer tf.keras when available
    from keras.layers import InputLayer as _TKInputLayer
except Exception:
    try:
        # fallback to standalone keras if present
        from keras.layers import InputLayer as _TKInputLayer
    except Exception:
        _TKInputLayer = None


class CompatInputLayer(_TKInputLayer if _TKInputLayer is not None else object):
    @classmethod
    def from_config(cls, config):
        # remap legacy 'batch_shape' to 'batch_input_shape' if present
        if "batch_shape" in config and "batch_input_shape" not in config:
            config["batch_input_shape"] = config.pop("batch_shape")
        # If we don't have a real InputLayer base (rare), just return config
        if _TKInputLayer is None:
            return config
        return super().from_config(config)


# =======================
# HARD-CODED ABSOLUTE PATHS
# =======================
RF_MODEL_PATH = r"D:\projects\organtrust\ml\artifacts\rf_model.pkl"
RF_SCALER_PATH = r"D:\projects\organtrust\ml\artifacts\scaler.pkl"

DEEP_MODEL_PATH = r"D:\projects\organtrust\ml\artifacts\deep_model.h5"  # change to .keras if needed
DEEP_SCALER_PATH = r"D:\projects\organtrust\ml\artifacts\deep_scaler.pkl"

FAISS_INDEX_PATH = r"D:\projects\organtrust\rag_kb\kb_index.faiss"
KB_ROWS_PATH = r"D:\projects\organtrust\rag_kb\kb_rows.csv"

HF_TOKEN = "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"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


# =======================
# LOAD MODELS
# =======================
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load(RF_MODEL_PATH)
        rf_scaler = joblib.load(RF_SCALER_PATH)

        # Try loading the deep model. Use compile=False to avoid issues with
        # missing optimizer state. Provide fallbacks for common deserialization
        # mismatches (InputLayer batch_shape and legacy DTypePolicy).
        try:
            deep_model = tf.keras.models.load_model(DEEP_MODEL_PATH, compile=False)
        except Exception as first_exc:
            # First fallback: remap legacy InputLayer keys
            try:
                deep_model = tf.keras.models.load_model(
                    DEEP_MODEL_PATH,
                    compile=False,
                    custom_objects={"InputLayer": CompatInputLayer},
                )
            except Exception as second_exc:
                # Second fallback: map legacy DTypePolicy to TF's Policy (if seen)
                second_msg = str(second_exc).lower()
                first_msg = str(first_exc).lower()
                if "dtypepolicy" in second_msg or "unknown dtype policy" in second_msg or "dtypepolicy" in first_msg:
                    try:
                        deep_model = tf.keras.models.load_model(
                            DEEP_MODEL_PATH,
                            compile=False,
                            custom_objects={
                                "InputLayer": CompatInputLayer,
                                "DTypePolicy": tf.keras.mixed_precision.Policy,
                            },
                        )
                    except Exception:
                        # If this also fails, raise the most informative error
                        raise second_exc
                else:
                    # re-raise the original error if it's unrelated
                    raise

        deep_scaler = joblib.load(DEEP_SCALER_PATH)

        return rf_model, rf_scaler, deep_model, deep_scaler
    except Exception as e:
        st.error(f"❌ Error loading ML/DL models: {e}")
        raise


rf_model, rf_scaler, deep_model, deep_scaler = load_models()


# =======================
# LOAD RAG COMPONENTS
# =======================
@st.cache_resource
def load_rag():
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        rows = pd.read_csv(KB_ROWS_PATH)
        embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)
        return index, rows, embedder, client
    except Exception as e:
        st.error(f"❌ Error loading RAG components: {e}")
        raise


index, rows, embedder, client = load_rag()


# =======================
# RAG HELPERS
# =======================
def retrieve_context(query, top_k=5):
    q_vec = embedder.encode([query], show_progress_bar=False).astype("float32")
    scores, ids = index.search(q_vec, top_k)
    snippets = [rows.iloc[int(i)]["snippet"] for i in ids[0] if i < len(rows)]
    return "\n\n".join(snippets)


def generate_answer(prompt):
    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.7,
            top_p=0.95,
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"⚠️ API Error: {e}"


def ask_llm(query):
    context = retrieve_context(query)
    if not context:
        return "⚠️ No relevant context found in Knowledge Base."
    prompt = f"""
You are a medical AI assistant specializing in organ transplant compatibility.
Use the context below to answer clearly and accurately.

Context:
{context}

Question:
{query}

Answer concisely:
"""
    return generate_answer(prompt)


# =======================
# STREAMLIT UI
# =======================
st.set_page_config(page_title="OrganTrust AI Platform", layout="wide")
st.title("🧠 OrganTrust AI Platform")
st.markdown("Empowering transplant prediction & intelligence through ML, DL, and RAG-based GenAI")

page = st.sidebar.radio("🔍 Choose Module", ["🏥 ML Model", "🧬 Deep Learning", "🧠 RAG Assistant"])


# =======================
# ML MODEL TAB
# =======================
if page == "🏥 ML Model":
    st.header("🩺 Random Forest Model – 1-Year Survival Prediction")

    features = [
        "donor_age",
        "donor_egfr_ml_min_1_73m2",
        "donor_creatinine_mg_dl",
        "recipient_age",
        "recipient_dialysis_months",
        "recipient_creatinine_mg_dl_pre_tx",
        "recipient_hemoglobin_g_dl",
        "recipient_hla_antibodies_count",
    ]

    user_input = {}
    cols = st.columns(4)
    for i, f in enumerate(features):
        with cols[i % 4]:
            user_input[f] = st.number_input(f.replace("_", " ").title(), value=0.0)

    if st.button("🔍 Predict (Random Forest)"):
        X = np.array([[user_input[f] for f in features]])
        X_scaled = rf_scaler.transform(X)
        pred_prob = rf_model.predict_proba(X_scaled)[0][1]
        st.success(f"🧾 Predicted 1-Year Survival Probability: **{pred_prob*100:.2f}%**")


# =======================
# DEEP LEARNING TAB
# =======================
elif page == "🧬 Deep Learning":
    st.header("🧬 Deep Neural Network – Survival Prediction")

    features = [
        "donor_age",
        "donor_egfr_ml_min_1_73m2",
        "donor_creatinine_mg_dl",
        "recipient_age",
        "recipient_dialysis_months",
        "recipient_creatinine_mg_dl_pre_tx",
        "recipient_hemoglobin_g_dl",
        "recipient_hla_antibodies_count",
    ]

    user_input = {}
    cols = st.columns(4)
    for i, f in enumerate(features):
        with cols[i % 4]:
            user_input[f] = st.number_input(f.replace("_", " ").title(), value=0.0)

    if st.button("⚙️ Predict (Deep Learning)"):
        X = np.array([[user_input[f] for f in features]])
        X_scaled = deep_scaler.transform(X)
        pred = deep_model.predict(X_scaled)[0][0]
        st.success(f"🧾 Predicted 1-Year Survival Probability: **{pred*100:.2f}%**")


# =======================
# RAG ASSISTANT TAB
# =======================
else:
    st.header("🧠 OrganTrust RAG Assistant")
    st.markdown("Ask questions about **organ transplant compatibility, donor health, or recipient conditions.**")

    user_query = st.text_input("💬 Type your question here:")

    if st.button("🚀 Ask OrganTrust AI"):
        with st.spinner("Generating response..."):
            start = time.time()
            answer = ask_llm(user_query)
            end = time.time()
        st.markdown(f"**🧠 Answer:**\n\n{answer}")
        st.caption(f"⏱️ Response time: {end - start:.2f}s")

st.sidebar.markdown("---")
