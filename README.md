# OrganTrust – AI-Powered Organ Transplant Compatibility & Decision Support System

OrganTrust is an end-to-end intelligent health-tech system that uses Machine Learning, Deep Learning, and Generative AI (RAG) to support doctors in evaluating donor–recipient kidney transplant compatibility.

## 🚀 Key Features

### 1. Donor–Recipient Compatibility Prediction (ML)
- Random Forest classifier  
- Predicts 0 or 1 → transplant compatibility  
- Accuracy: ~82–87%  
- Uses 8 structured medical parameters  

### 2. 1-Year Survival Probability (Deep Learning)
- Dense neural network  
- BatchNorm + Dropout  
- Outputs probability between 0–1  
- Trained on clinical indicators  

### 3. RAG-Powered Medical Assistant (Gen-AI)
- FAISS vector database  
- SentenceTransformer bge-small-en-v1.5 embeddings  
- Hugging Face Inference API  
- Llama-3.1 Chat model  
- Provides safe, medically aligned responses  

### 4. Unified Streamlit Application
- Compatibility ML prediction  
- Survival DL prediction  
- RAG-powered medical Q&A  
- Real-time inference + chat history  

## 🧠 System Architecture

```
             ┌────────────────────────────┐
             │      User Interface        │
             │        (Streamlit)         │
             └──────────────┬─────────────┘
                            │
      ┌─────────────────────┼────────────────────────┐
      │                     │                        │
┌─────────────┐     ┌─────────────┐       ┌──────────────────┐
│ ML Module    │     │ DL Module    │       │ RAG Module       │
│ RandomForest │     │ Deep Model   │       │ Llama-3.1 Chat   │
└───────┬──────┘     └──────┬──────┘       └──────┬───────────┘
        │                   │                      │
   Scaler.pkl         deep_scaler.pkl       FAISS Index + Embeddings
   rf_model.pkl       deep_model.h5         SentenceTransformer
                                              HuggingFace API
```

## 📂 Project Structure

```
organtrust/
│
├── data/
│   └── paired_data.csv
│
├── ml/
│   ├── train_rf.py
│   ├── train_deep.py
│   └── artifacts/
│       ├── scaler.pkl
│       ├── rf_model.pkl
│       ├── deep_model.h5
│       └── deep_scaler.pkl
│
├── rag/
│   ├── build_kb.py
│   ├── rag_pipeline_llama_fixed.py
│   └── rag_kb/
│       ├── kb_index.faiss
│       └── kb_rows.csv
│
├── frontend/
│   └── app.py
│
├── requirements.txt
└── README.md
```

## ⚙️ Installation

### 1. Clone Repository
```
git clone https://github.com/anupam2607/organtrust.git
cd organtrust
```

### 2. Create Virtual Environment
```
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

## 🧬 Train ML & DL Models

### Train Random Forest Model
```
python ml/train_rf.py
```

### Train Deep Learning Model
```
python ml/train_deep.py
```

Artifacts saved under:
```
ml/artifacts/
```

## 🔍 Build RAG Knowledge Base
```
python rag/build_kb.py
```

Outputs FAISS index + KB metadata inside:
```
rag/rag_kb/
```

## 🤖 Run RAG Assistant (CLI)
```
python rag/rag_pipeline_llama_fixed.py
```

## 🌐 Run Streamlit App
```
streamlit run frontend/app.py
```

App URL:
```
http://localhost:8501/
```

## 📊 Model Inputs (ML & DL)

| Feature | Description |
|--------|-------------|
| donor_age | Donor age |
| donor_egfr_ml_min_1_73m2 | Kidney filtration rate |
| donor_creatinine_mg_dl | Donor creatinine |
| recipient_age | Recipient age |
| recipient_dialysis_months | Dialysis duration |
| recipient_creatinine_mg_dl_pre_tx | Pre-transplant creatinine |
| recipient_hemoglobin_g_dl | Hemoglobin level |
| recipient_hla_antibodies_count | HLA antibodies count |
