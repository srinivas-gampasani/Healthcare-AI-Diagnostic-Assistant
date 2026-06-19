"""
Healthcare AI Diagnostic Assistant — FastAPI Server
====================================================
HIPAA-compliant REST API with JWT authentication.

Endpoints:
  POST /auth/token              → Login, get JWT
  GET  /patients                → List patients
  GET  /patients/{id}           → Patient details (FHIR)
  GET  /patients/{id}/summary   → Clinical summary
  POST /query                   → Ask clinical question (AI)
  POST /patients/{id}/note      → Generate draft clinical note
  GET  /patients/{id}/alerts    → Critical value alerts
  GET  /audit                   → HIPAA audit log (admin only)
  GET  /api/health              → Health check
"""
import logging
from datetime import timedelta
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import settings
from core.auth import authenticate_user, create_access_token, decode_token, has_permission, DEMO_USERS
from core.hipaa import audit_logger, phi_masker
from ehr.fhir_models import ehr_db
from agents.clinical_assistant import assistant

import os

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL),
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Healthcare AI Diagnostic Assistant",
    description="HIPAA-compliant clinical AI assistant with GPT-4 + RAG over EHR data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


# ── Auth helpers ───────────────────────────────────────────────────────────────

async def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    username = payload.get("sub")
    user = DEMO_USERS.get(username)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def require_permission(permission: str):
    async def checker(user: dict = Depends(get_current_user)):
        if not has_permission(user["role"], permission):
            raise HTTPException(status_code=403, detail=f"Permission '{permission}' required. Your role: {user['role']}")
        return user
    return checker


# ── Pydantic models ────────────────────────────────────────────────────────────

class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict


class ClinicalQueryRequest(BaseModel):
    question: str
    patient_id: Optional[str] = None
    session_id: Optional[str] = None
    include_rag: bool = True


class ClinicalQueryResponse(BaseModel):
    answer: str
    critical_alerts: List[str]
    patient_context_used: bool
    processing_time_ms: int
    phi_detected: List[str]
    session_id: Optional[str]


class NoteRequest(BaseModel):
    note_type: str = "progress"


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "hipaa_compliant": True,
        "phi_masking": settings.PHI_MASKING_ENABLED,
        "model": settings.OPENAI_MODEL,
        "patients_loaded": len(ehr_db.list_all_patients()),
    }


@app.post("/auth/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    audit_logger.log(user["username"], "login", "auth", "system", outcome="success")
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {k: v for k, v in user.items() if k != "hashed_password"},
    }


@app.get("/patients")
async def list_patients(user: dict = Depends(require_permission("view_patient"))):
    patients = ehr_db.list_all_patients()
    audit_logger.log(user["username"], "list_patients", "patient", "all")
    return [
        {
            "id": p.id,
            "mrn": p.mrn,
            "name": p.full_name,
            "age": p.age,
            "gender": p.gender,
            "primary_physician": p.primary_physician,
            "active_conditions": len(ehr_db.get_active_conditions(p.id)),
            "active_medications": len(ehr_db.get_medications(p.id)),
            "critical_alerts": len([o for o in ehr_db.get_abnormal_observations(p.id) if "critical" in o.interpretation]),
        }
        for p in patients
    ]


@app.get("/patients/{patient_id}")
async def get_patient(patient_id: str, user: dict = Depends(require_permission("view_patient"))):
    patient = ehr_db.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    audit_logger.log(user["username"], "view_patient", "patient", patient_id)
    return {
        "fhir": patient.to_fhir_dict(),
        "conditions": [{"code": c.icd10_code, "display": c.display, "status": c.clinical_status, "onset": c.onset_date}
                       for c in ehr_db.get_active_conditions(patient_id)],
        "medications": [{"name": m.medication_name, "dosage": m.dosage, "frequency": m.frequency,
                         "route": m.route, "indication": m.indication}
                        for m in ehr_db.get_medications(patient_id)],
        "recent_observations": [{"display": o.display, "value": o.value, "unit": o.unit,
                                   "interpretation": o.interpretation, "date": o.effective_date}
                                  for o in ehr_db.get_observations(patient_id)[:10]],
    }


@app.get("/patients/{patient_id}/summary")
async def patient_summary(patient_id: str, user: dict = Depends(require_permission("view_patient"))):
    summary = ehr_db.get_full_summary(patient_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Patient not found")
    audit_logger.log(user["username"], "view_summary", "patient", patient_id)
    return summary


@app.get("/patients/{patient_id}/alerts")
async def get_patient_alerts(patient_id: str, user: dict = Depends(require_permission("view_labs"))):
    alerts = assistant._check_critical_alerts(patient_id)
    audit_logger.log(user["username"], "view_alerts", "patient", patient_id)
    abnormal = ehr_db.get_abnormal_observations(patient_id)
    return {
        "patient_id": patient_id,
        "critical_alerts": alerts,
        "abnormal_values": [
            {"display": o.display, "value": o.value, "unit": o.unit,
             "interpretation": o.interpretation, "date": o.effective_date}
            for o in abnormal
        ],
        "alert_count": len(alerts),
    }


@app.post("/query", response_model=ClinicalQueryResponse)
async def clinical_query(req: ClinicalQueryRequest, user: dict = Depends(require_permission("query"))):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = assistant.query(
        question=req.question,
        patient_id=req.patient_id,
        user_id=user["username"],
        include_rag=req.include_rag,
    )

    return ClinicalQueryResponse(
        answer=result["answer"],
        critical_alerts=result["critical_alerts"],
        patient_context_used=result["patient_context_used"],
        processing_time_ms=result["processing_time_ms"],
        phi_detected=result.get("phi_detected_in_query", []),
        session_id=req.session_id,
    )


@app.post("/patients/{patient_id}/note")
async def generate_note(patient_id: str, req: NoteRequest, user: dict = Depends(require_permission("generate_note"))):
    patient = ehr_db.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    audit_logger.log(user["username"], "generate_note", "patient", patient_id, details={"note_type": req.note_type})
    note = assistant.generate_clinical_note(patient_id, req.note_type, user["username"])
    return {"patient_id": patient_id, "note_type": req.note_type, "draft": note,
            "generated_by": "Healthcare AI v1.0", "requires_attestation": True}


@app.get("/patients/{patient_id}/notes")
async def get_notes(patient_id: str, user: dict = Depends(require_permission("view_patient"))):
    notes = ehr_db.get_notes(patient_id)
    return [{"id": n.id, "type": n.note_type, "author": n.author, "date": n.date,
             "chief_complaint": n.chief_complaint, "assessment": n.assessment[:200]}
            for n in notes]


@app.get("/audit")
async def get_audit_log(user: dict = Depends(require_permission("view_audit"))):
    entries = audit_logger.get_recent(100)
    return {"entries": entries, "total": len(entries)}


# ── Static UI ──────────────────────────────────────────────────────────────────
ui_path = os.path.join(os.path.dirname(__file__), "..", "ui")
if os.path.exists(ui_path):
    app.mount("/", StaticFiles(directory=ui_path, html=True), name="ui")
