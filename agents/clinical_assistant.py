"""
Clinical AI Assistant
=====================
Core AI engine combining:
- GPT-4 for reasoning and generation
- RAG over clinical guidelines
- EHR patient context (FHIR)
- HIPAA-compliant PHI handling
- Clinical note drafting
- Drug interaction checking
- Clinical decision support alerts
"""
import json
import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, Any, AsyncGenerator

from config import settings
from core.hipaa import phi_masker, audit_logger
from ehr.fhir_models import ehr_db, FHIRPatient
from rag.engine import rag_engine

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a HIPAA-compliant Clinical AI Assistant deployed at Ascension hospital network.

ROLE:
You assist physicians, nurses, and care coordinators by:
1. Answering clinical questions grounded in evidence-based guidelines (ACC/AHA, ADA, KDIGO, GINA)
2. Summarizing patient history, labs, medications, and clinical notes from the EHR
3. Generating draft clinical summaries and SOAP notes (marked [AI-DRAFT] — require physician review)
4. Surfacing relevant clinical decision support alerts (critical lab values, drug interactions)
5. Providing differential diagnosis assistance

CRITICAL GUIDELINES:
- SAFETY FIRST: Always flag critical lab values and urgent clinical situations prominently
- ACCURACY: Base responses on the provided clinical context and guidelines. Do not fabricate lab values or medications
- UNCERTAINTY: If information is missing or unclear, say so explicitly
- SCOPE: You assist clinical decision-making; you do NOT replace physician judgment
- HIPAA: Do not repeat patient identifiers unnecessarily. Focus on clinical content
- DISCLAIMERS: AI-generated clinical summaries must be labeled [AI-DRAFT] and require physician attestation

RESPONSE FORMAT:
- Use markdown headers and bullet points for clinical content
- Highlight ⚠️ CRITICAL ALERTS in bold for urgent findings
- Use [AI-DRAFT] prefix for generated clinical documents
- Cite guidelines when making recommendations (e.g., "per ADA 2024 Standards")"""


class ClinicalAIAssistant:
    """
    Main clinical AI assistant class.
    Orchestrates EHR lookup, RAG retrieval, and LLM generation.
    """

    def __init__(self):
        self.model = settings.OPENAI_MODEL
        self._client = None

    def _get_client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        return self._client

    def _build_patient_context(self, patient_id: str) -> str:
        """Build comprehensive patient context string from EHR."""
        patient = ehr_db.get_patient(patient_id)
        if not patient:
            return f"Patient {patient_id} not found in EHR."

        conditions = ehr_db.get_active_conditions(patient_id)
        medications = ehr_db.get_medications(patient_id, active_only=True)
        observations = ehr_db.get_observations(patient_id)[:8]
        abnormal = ehr_db.get_abnormal_observations(patient_id)
        notes = ehr_db.get_notes(patient_id)[:1]

        lines = [
            f"## Patient: {patient.full_name} | Age {patient.age} | {patient.gender.upper()}",
            f"MRN: {patient.mrn} | DOB: {patient.date_of_birth} | Blood Type: {patient.blood_type}",
            f"PCP: {patient.primary_physician} | Allergies: {', '.join(patient.allergies) or 'NKDA'}",
            "",
            "### Active Diagnoses:",
        ]
        for c in conditions:
            lines.append(f"- {c.display} ({c.icd10_code}) — {c.clinical_status}, onset {c.onset_date}")

        lines.append("\n### Active Medications:")
        for m in medications:
            lines.append(f"- {m.medication_name} {m.dosage} {m.frequency} [{m.route}] — for {m.indication}")

        if abnormal:
            lines.append("\n### ⚠️ ABNORMAL LAB/VITAL VALUES:")
            for o in abnormal:
                flag = "🔴 CRITICAL" if "critical" in o.interpretation else "🟡 ABNORMAL"
                lines.append(f"- {flag} {o.display}: {o.value} {o.unit} ({o.interpretation}) — {o.effective_date}")

        lines.append("\n### Recent Lab/Vital Values:")
        for o in observations[:6]:
            lines.append(f"- {o.display}: {o.value} {o.unit} ({o.interpretation}) — {o.effective_date}")

        if notes:
            n = notes[0]
            lines.append(f"\n### Most Recent Clinical Note ({n.date} — {n.note_type}):")
            lines.append(f"**Chief Complaint:** {n.chief_complaint}")
            lines.append(f"**Assessment:** {n.assessment}")
            lines.append(f"**Plan:** {n.plan}")

        return "\n".join(lines)

    def _retrieve_clinical_context(self, query: str, patient_id: Optional[str] = None) -> str:
        """Retrieve relevant clinical guidelines via RAG."""
        rag_engine.initialize()

        icd10_boost = []
        if patient_id:
            conditions = ehr_db.get_active_conditions(patient_id)
            icd10_boost = [c.icd10_code for c in conditions]

        chunks = rag_engine.retrieve(query, top_k=4, icd10_boost=icd10_boost if icd10_boost else None)
        if not chunks:
            return ""
        return rag_engine.format_context(chunks, max_chars=5000)

    def _check_critical_alerts(self, patient_id: str) -> List[str]:
        """Generate critical clinical alerts for a patient."""
        alerts = []
        abnormal = ehr_db.get_abnormal_observations(patient_id)

        for obs in abnormal:
            if "critical" in obs.interpretation:
                alerts.append(
                    f"⚠️ CRITICAL: {obs.display} = {obs.value} {obs.unit} "
                    f"({obs.interpretation}) — requires immediate attention"
                )

        # Drug interaction checks
        meds = ehr_db.get_medications(patient_id)
        med_names = {m.medication_name.lower() for m in meds}
        conditions = ehr_db.get_active_conditions(patient_id)
        condition_codes = {c.icd10_code for c in conditions}

        if "N18.3" in condition_codes or "N18.4" in condition_codes:
            if any("metformin" in m for m in med_names):
                alerts.append("⚠️ DRUG ALERT: Metformin in CKD — verify eGFR ≥30 before continuing")
            if any("nsaid" in m or "ibuprofen" in m or "naproxen" in m for m in med_names):
                alerts.append("⚠️ DRUG ALERT: NSAID use in CKD — avoid (nephrotoxic)")

        if "I50.9" in condition_codes or "I50.2" in condition_codes:
            if any("amlodipine" in m for m in med_names):
                alerts.append("⚠️ DRUG ALERT: Non-DHP CCB caution in HFrEF")

        return alerts

    def query(
        self,
        question: str,
        patient_id: Optional[str] = None,
        user_id: str = "clinician",
        include_rag: bool = True,
        session_history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Main query endpoint.
        
        Returns:
            {
                answer: str,
                critical_alerts: List[str],
                retrieved_sources: List[str],
                patient_context_used: bool,
                processing_time_ms: int,
            }
        """
        import time
        t0 = time.time()

        # Audit log
        audit_logger.log(
            user_id=user_id,
            action="clinical_query",
            resource_type="patient" if patient_id else "general",
            resource_id=patient_id or "N/A",
            details={"query_length": len(question)},
        )

        # PHI check on incoming query
        masked_q, phi_found = phi_masker.mask(question)
        if phi_found:
            logger.warning(f"PHI detected in query ({phi_found}) — masked before processing")

        # Build context
        patient_context = ""
        critical_alerts = []
        if patient_id:
            patient_context = self._build_patient_context(patient_id)
            critical_alerts = self._check_critical_alerts(patient_id)

        guideline_context = ""
        if include_rag:
            guideline_context = self._retrieve_clinical_context(masked_q, patient_id)

        # Build messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if session_history:
            messages.extend(session_history[-6:])  # last 3 turns

        # Compose user message
        user_content_parts = []

        if critical_alerts:
            user_content_parts.append("## ⚠️ ACTIVE CRITICAL ALERTS:\n" + "\n".join(critical_alerts))

        if patient_context:
            user_content_parts.append(f"## EHR PATIENT CONTEXT:\n{patient_context}")

        if guideline_context:
            user_content_parts.append(f"## RELEVANT CLINICAL GUIDELINES (from RAG):\n{guideline_context}")

        user_content_parts.append(f"## CLINICAL QUESTION:\n{masked_q}")

        messages.append({"role": "user", "content": "\n\n".join(user_content_parts)})

        # LLM call
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,         # Low temp for clinical accuracy
                max_tokens=2000,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            answer = self._fallback_response(masked_q, patient_id, critical_alerts)

        elapsed_ms = int((time.time() - t0) * 1000)

        return {
            "answer": answer,
            "critical_alerts": critical_alerts,
            "retrieved_sources": [f"RAG: {len(guideline_context.split())} guideline words"] if guideline_context else [],
            "patient_context_used": bool(patient_context),
            "processing_time_ms": elapsed_ms,
            "phi_detected_in_query": phi_found,
        }

    def _fallback_response(
        self, question: str, patient_id: Optional[str], alerts: List[str]
    ) -> str:
        """
        Offline/fallback response when LLM API unavailable.
        Still returns patient context and alerts.
        """
        parts = [f"**[OFFLINE MODE — OpenAI API unavailable]**\n"]

        if alerts:
            parts.append("## ⚠️ Critical Alerts:\n" + "\n".join(alerts))

        if patient_id:
            patient = ehr_db.get_patient(patient_id)
            if patient:
                conditions = ehr_db.get_active_conditions(patient_id)
                meds = ehr_db.get_medications(patient_id)
                abnormal = ehr_db.get_abnormal_observations(patient_id)

                parts.append(f"\n## Patient Summary: {patient.full_name}, Age {patient.age}")
                parts.append(f"**Active Conditions:** {', '.join(c.display for c in conditions)}")
                parts.append(f"**Medications:** {', '.join(m.medication_name for m in meds)}")
                if abnormal:
                    parts.append(f"**Abnormal Values:** {', '.join(f'{o.display}={o.value}{o.unit}' for o in abnormal)}")

        parts.append(f"\n**Your Question:** {question}")
        parts.append("\n*Note: AI reasoning unavailable — please consult clinical references directly.*")
        return "\n\n".join(parts)

    def generate_clinical_note(
        self, patient_id: str, note_type: str = "progress", user_id: str = "clinician"
    ) -> str:
        """Generate a draft clinical note (SOAP format) from EHR data."""
        audit_logger.log(user_id, "generate_note", "patient", patient_id, details={"note_type": note_type})

        patient_context = self._build_patient_context(patient_id)
        patient = ehr_db.get_patient(patient_id)
        existing_notes = ehr_db.get_notes(patient_id)

        prompt = f"""Generate a detailed draft {note_type} note in SOAP format for this patient.

{patient_context}

{"Most recent note for reference: " + existing_notes[0].plan if existing_notes else ""}

Requirements:
- SOAP format (Subjective / Objective / Assessment / Plan)
- Base assessment on actual lab values and conditions shown
- Plan should follow current evidence-based guidelines
- Start with: [AI-DRAFT — REQUIRES PHYSICIAN REVIEW AND ATTESTATION]
- Date: {datetime.now().strftime('%Y-%m-%d')}
"""
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.15,
                max_tokens=1500,
            )
            return response.choices[0].message.content
        except Exception as e:
            return self._fallback_note(patient_id, note_type, patient_context)

    def _fallback_note(self, patient_id: str, note_type: str, context: str) -> str:
        patient = ehr_db.get_patient(patient_id)
        if not patient:
            return "[AI-DRAFT] Patient not found."

        conditions = ehr_db.get_active_conditions(patient_id)
        meds = ehr_db.get_medications(patient_id)
        abnormal = ehr_db.get_abnormal_observations(patient_id)

        return f"""[AI-DRAFT — REQUIRES PHYSICIAN REVIEW AND ATTESTATION]

**{note_type.upper()} NOTE**
**Date:** {datetime.now().strftime('%Y-%m-%d')}
**Patient:** {patient.full_name}, Age {patient.age}, {patient.gender}

**SUBJECTIVE:**
Patient presents for {note_type} evaluation. Active conditions include {', '.join(c.display for c in conditions[:3])}.

**OBJECTIVE:**
{"Abnormal values: " + "; ".join(f"{o.display} {o.value}{o.unit}" for o in abnormal) if abnormal else "Vitals and labs within acceptable ranges."}

**ASSESSMENT:**
{chr(10).join(f"{i+1}. {c.display} ({c.icd10_code}) — {c.clinical_status}" for i, c in enumerate(conditions))}

**PLAN:**
{chr(10).join(f"• Continue {m.medication_name} {m.dosage} {m.frequency}" for m in meds[:5])}
• Follow up per clinical judgment.

*[AI-generated draft — physician must review, edit, and attest before finalizing]*
"""


# Singleton
assistant = ClinicalAIAssistant()
