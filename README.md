# 👗 AI-Powered Personal Fashion Stylist for E-Commerce Using RAG

> **B.Tech Final Year Project** | AI/ML + GenAI + RAG Developer: Sparsh

---

## 🧠 Project Overview

This system provides personalized fashion recommendations by:
1. Analyzing user photos (face shape, skin tone, body type) using **Gemini Vision API**
2. Retrieving relevant fashion rules from a **ChromaDB** vector database (RAG)
3. Generating final recommendations using **Gemini LLM**
4. Exposing everything via a **FastAPI** REST endpoint

---

## 🗂️ Project Structure (AI Layer)

```
AI-Powered Personal Fashion Stylist for E-Commerce Using RAG/
│
├── app/                        # FastAPI application
│   ├── main.py                 # FastAPI entry point
│   ├── routers/
│   │   └── stylist.py          # API route definitions
│   └── schemas.py              # Pydantic request/response models
│
├── ai_engine/                  # Core AI logic (YOUR main work)
│   ├── vision_analyzer.py      # Gemini Vision API - photo analysis
│   ├── rag_pipeline.py         # ChromaDB + retrieval logic
│   ├── recommender.py          # Final recommendation generator
│   └── prompt_templates.py     # All Gemini prompts (versioned)
│
├── knowledge_base/             # Fashion rules for RAG
│   ├── builder.py              # Script to build/populate ChromaDB
│   └── fashion_rules.py        # Fashion knowledge documents
│
├── evaluation/                 # Testing & accuracy metrics
│   ├── test_vision.py          # Test Gemini Vision accuracy
│   ├── test_rag.py             # Test retrieval quality
│   └── results/                # Store evaluation results here
│
├── notebooks/                  # Jupyter notebooks for experiments
│   └── prompt_engineering.ipynb
│
├── chroma_db/                  # ChromaDB persistent storage (auto-created)
│
├── .env                        # API keys (never commit this!)
├── .env.example                # Template for environment variables
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🔄 System Flow

```
User Photo (base64 or file)
        ↓
[ Gemini Vision API ]
  → face_shape, skin_tone, body_type (JSON)
        ↓
[ ChromaDB RAG Search ]
  → Retrieve top-k fashion rules matching user profile
        ↓
[ Gemini LLM ]
  → Combine analysis + retrieved rules → final recommendation
        ↓
[ FastAPI Response ]
  → Return structured JSON to frontend/backend team
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up API key
```bash
cp .env.example .env
# Edit .env and add your Google AI Studio API key
```

### 3. Build the knowledge base (one-time)
```bash
python knowledge_base/builder.py
```

### 4. Run the API server
```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Test the API
Visit: `http://localhost:8000/docs` (Swagger UI)

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| AI Model | Google Gemini 1.5 Flash (Vision + LLM) |
| Embeddings | Gemini Embedding API (`text-embedding-004`) |
| Vector DB | ChromaDB (local persistent) |
| API Framework | FastAPI + Uvicorn |
| Validation | Pydantic v2 |

---

## 📊 Evaluation Metrics

- **Face Shape Detection Accuracy**: % correct classifications
- **Skin Tone Detection**: Qualitative + quantitative assessment
- **RAG Retrieval Precision**: Are retrieved rules relevant?
- **Recommendation Quality**: Human evaluation rubric

---

## 📝 Viva Key Points

1. **Why RAG?** Grounds Gemini's responses in curated fashion domain knowledge, prevents hallucinations
2. **Why ChromaDB?** Lightweight, local, no cloud cost — perfect for academic project
3. **Why Gemini?** Free tier via Google AI Studio, multimodal (vision + text)
4. **Prompt Engineering**: See `ai_engine/prompt_templates.py` for all versioned prompts
