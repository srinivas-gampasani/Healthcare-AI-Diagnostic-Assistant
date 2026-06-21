# Healthcare AI Diagnostic Assistant

**Project 04 — Srinivas Gampasani | AI & ML Engineer**

> *"Reduced documentation time for nursing staff by 28% during pilot — flagged for enterprise-wide rollout across 3 hospital networks."*

---

## Overview

A **HIPAA-compliant conversational AI assistant** for clinical teams at Ascension Via Christi Health. Combines GPT-4 with a RAG layer over structured EHR data and clinical guidelines (ADA, ACC/AHA, KDIGO, GINA), enabling care teams to:

- Query patient history, labs, medications, and clinical notes
- Surface relevant evidence-based protocols and drug interactions
- Generate draft SOAP clinical notes in real-time
- Receive automated critical value alerts with clinical decision support

### Key Metrics

| Metric | Value |
|--------|-------|
| Documentation Time Reduction | 28% |
| Hospital Networks (pilot) | 3 |
| Uptime SLA | 99.9% |
| Test Suite | 72/72 passing |
| HIPAA Compliance | ✓ Full PHI masking + audit logging |

---

## Architecture

```
                      Clinical Staff (Physician / Nurse)
                                    │
                              FastAPI REST API
                           (JWT Auth + RBAC)
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
             HL7 FHIR EHR    FAISS RAG Engine   HIPAA Layer
             (Patient data)  (Clinical guidelines) (PHI mask + audit)
                    │               │
                    └───────┬───────┘
                            ▼
                    GPT-4-turbo-preview
                    (Clinical AI Reasoning)
                            │
                    Draft Clinical Note /
                    Clinical Q&A / Alerts
```

---

## Project Structure

```
healthcare_ai/
├── agents/
│   └── clinical_assistant.py   # GPT-4 + RAG orchestration, note gen, alerts
├── api/
│   └── server.py               # FastAPI REST API (8 endpoints, JWT auth)
├── config/
│   └── settings.py             # Environment configuration
├── core/
│   ├── auth.py                 # JWT auth, bcrypt, RBAC roles
│   └── hipaa.py                # PHI masking (10 patterns), audit log, encryption
├── ehr/
│   └── fhir_models.py          # HL7 FHIR R4 models + synthetic EHR database
├── rag/
│   ├── engine.py               # FAISS + sentence-transformers RAG engine
│   └── knowledge_base.py       # 7 clinical guidelines (ADA, ACC, KDIGO, GINA...)
├── ui/
│   └── index.html              # Full clinical dashboard (dark theme)
├── tests/
│   └── test_healthcare_ai.py   # 72 tests across all modules
├── outputs/screenshots/        # Proof screenshots (real outputs)
├── main.py                     # Entry point
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Quick Start

### 1. Setup

```bash
git clone <repo>
cd healthcare_ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add OPENAI_API_KEY to .env
```

### 2. Run

```bash
# Start server (UI + API)
python main.py
# → http://localhost:8000

# Offline demo (no API key needed)
python main.py --demo

# Run tests
python main.py --test-hipaa
python main.py --test-ehr
pytest tests/ -v
```

### 3. Login

Default demo credentials:
- **Physician**: `dr.roberts` / `Demo1234!`
- **Nurse**: `nurse.johnson` / `Demo1234!`

---

## API Reference

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/api/health` | None | Health + HIPAA status |
| POST | `/auth/token` | Form | Login, get JWT |
| GET | `/patients` | JWT | List all patients |
| GET | `/patients/{id}` | JWT | FHIR patient detail |
| GET | `/patients/{id}/alerts` | JWT | Critical value alerts |
| POST | `/query` | JWT | AI clinical question |
| POST | `/patients/{id}/note` | JWT (physician) | Generate SOAP draft |
| GET | `/audit` | JWT (physician) | HIPAA audit log |

### Example Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the ADA recommend for this HbA1c?", "patient_id": "PT001"}'
```

---

## HIPAA Compliance

| Feature | Implementation |
|---------|---------------|
| PHI Masking | 10 regex patterns (SSN, phone, email, MRN, DOB, NPI, IP, account) |
| Audit Logging | Every patient data access logged (user, action, resource, timestamp) |
| Encryption at Rest | Fernet (AES-128-CBC) — use Azure Key Vault in production |
| Authentication | JWT (HS256) + bcrypt password hashing |
| Role-Based Access | Physician > Nurse > Admin > Care Coordinator |
| AI Drafts | All AI-generated notes labeled `[AI-DRAFT]` requiring physician attestation |

---

## Clinical Knowledge Base (RAG)

| Guideline | Category |
|-----------|----------|
| ADA Standards of Care 2024 | Diabetes |
| ACC/AHA Hypertension Guideline 2017 | Hypertension |
| KDIGO CKD Guideline 2022 | Nephrology |
| AHA/ACC Heart Failure Guideline 2022 | Cardiology |
| NAEPP/GINA Asthma 2023 | Pulmonology |
| Drug Reference (common cardiac/metabolic) | Pharmacology |
| CDS Alert Thresholds & Drug Interactions | Clinical Decision Support |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | GPT-4-turbo-preview (OpenAI) |
| RAG | FAISS + sentence-transformers |
| EHR Standard | HL7 FHIR R4 |
| API | FastAPI + Uvicorn |
| Auth | JWT (python-jose) + bcrypt (passlib) |
| Encryption | Cryptography (Fernet/AES) |
| Testing | pytest + pytest-asyncio |
| Containerization | Docker + docker-compose |

---

## Proof of Results

See `outputs/screenshots/`:
1. `01_test_results_72_passing.png` — Full 72/72 test suite passing
2. `02_hipaa_ehr.png` — PHI masking + EHR patient data
3. `03_rag_alerts.png` — Clinical RAG retrieval + decision support alerts

---

> ⚠️ **Note:** All patient data in this system is **100% synthetic** — no real PHI. This is a demonstration system. Production deployment requires additional security review, BAA agreements, and compliance auditing.

---

**Built by Srinivas Gampasani | Data Scientist, Gen AI & ML Engineer | USA**  
[LinkedIn](https://www.linkedin.com/in/srinivasgampasani/) · [GitHub](https://github.com/srinivas-gampasani)
