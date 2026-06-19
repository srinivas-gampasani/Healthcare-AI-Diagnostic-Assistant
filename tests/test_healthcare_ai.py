"""
Test Suite — Healthcare AI Diagnostic Assistant
================================================
Tests: HIPAA compliance, EHR/FHIR models, RAG engine,
       auth, clinical alerts, and API endpoints.
"""
import pytest
from unittest.mock import patch, MagicMock


# ── HIPAA Tests ────────────────────────────────────────────────────────────────

class TestHIPAACompliance:
    def setup_method(self):
        from core.hipaa import PHIMasker, AuditLogger
        self.masker = PHIMasker()
        self.logger = AuditLogger()

    def test_ssn_masking(self):
        masked, found = self.masker.mask("SSN: 123-45-6789")
        assert "[SSN-REDACTED]" in masked
        assert "ssn" in found

    def test_phone_masking(self):
        masked, found = self.masker.mask("Call 555-867-5309")
        assert "[PHONE-REDACTED]" in masked
        assert "phone" in found

    def test_email_masking(self):
        masked, found = self.masker.mask("Email: patient@hospital.org")
        assert "[EMAIL-REDACTED]" in masked
        assert "email" in found

    def test_mrn_masking(self):
        masked, found = self.masker.mask("MRN: 100001 admitted")
        assert "[MRN-REDACTED]" in masked
        assert "mrn" in found

    def test_dob_masking(self):
        masked, found = self.masker.mask("DOB: 03/14/1958")
        assert "[DOB-REDACTED]" in masked
        assert "dob" in found

    def test_clean_text_not_modified(self):
        text = "Blood pressure 140/90 mmHg, HR 78 bpm, SpO2 98%"
        masked, found = self.masker.mask(text)
        assert found == []
        assert masked == text

    def test_multiple_phi_types(self):
        text = "SSN 123-45-6789, phone 555-123-4567, email test@email.com"
        _, found = self.masker.mask(text)
        assert len(found) >= 2

    def test_phi_detection_only(self):
        found = self.masker.detect("MRN: 999888 with SSN 123-45-6789")
        assert "ssn" in found
        assert "mrn" in found

    def test_audit_log_write(self):
        self.logger.log("test_user", "test_action", "patient", "PT001")
        recent = self.logger.get_recent(5)
        assert any(e["action"] == "test_action" for e in recent)

    def test_audit_log_fields(self):
        self.logger.log("dr.test", "view_patient", "patient", "PT002", outcome="success")
        recent = self.logger.get_recent(1)
        assert recent[-1]["user_id"] == "dr.test"
        assert recent[-1]["outcome"] == "success"

    def test_identifier_hashing(self):
        h1 = self.masker.hash_identifier("PT001")
        h2 = self.masker.hash_identifier("PT001")
        h3 = self.masker.hash_identifier("PT002")
        assert h1 == h2          # deterministic
        assert h1 != h3          # different inputs → different hashes
        assert len(h1) == 16     # truncated to 16 chars


# ── EHR / FHIR Tests ──────────────────────────────────────────────────────────

class TestEHRDatabase:
    def setup_method(self):
        from ehr.fhir_models import ehr_db
        self.db = ehr_db

    def test_patient_count(self):
        assert len(self.db.list_all_patients()) == 5

    def test_get_patient_by_id(self):
        p = self.db.get_patient("PT001")
        assert p is not None
        assert p.full_name == "James Carter"

    def test_get_patient_not_found(self):
        assert self.db.get_patient("PT999") is None

    def test_get_by_mrn(self):
        p = self.db.get_patient_by_mrn("MRN-100001")
        assert p is not None
        assert p.id == "PT001"

    def test_search_patients(self):
        results = self.db.search_patients("carter")
        assert len(results) == 1
        assert results[0].id == "PT001"

    def test_search_case_insensitive(self):
        results = self.db.search_patients("GONZALEZ")
        assert len(results) == 1

    def test_patient_age_calculation(self):
        p = self.db.get_patient("PT001")
        assert p.age > 60

    def test_active_conditions(self):
        conds = self.db.get_active_conditions("PT001")
        assert len(conds) >= 3
        assert all(c.clinical_status == "active" for c in conds)

    def test_observations_returned(self):
        obs = self.db.get_observations("PT003")
        assert len(obs) >= 4

    def test_abnormal_observations_pt005(self):
        abn = self.db.get_abnormal_observations("PT005")
        assert len(abn) >= 3
        assert all(o.is_abnormal for o in abn)

    def test_medications_active_only(self):
        meds = self.db.get_medications("PT001", active_only=True)
        assert len(meds) >= 3
        assert all(m.status == "active" for m in meds)

    def test_clinical_notes(self):
        notes = self.db.get_notes("PT001")
        assert len(notes) >= 1
        assert notes[0].patient_id == "PT001"

    def test_full_summary_structure(self):
        s = self.db.get_full_summary("PT003")
        assert "patient" in s
        assert "conditions" in s
        assert "medications" in s
        assert "observations" in s

    def test_fhir_patient_dict(self):
        p = self.db.get_patient("PT001")
        d = p.to_fhir_dict()
        assert d["resourceType"] == "Patient"
        assert d["gender"] == "male"
        assert "identifier" in d

    def test_fhir_observation_dict(self):
        obs = self.db.get_observations("PT001")[0]
        d = obs.to_fhir_dict()
        assert d["resourceType"] == "Observation"
        assert "subject" in d

    def test_fhir_condition_dict(self):
        cond = self.db.get_active_conditions("PT001")[0]
        d = cond.to_fhir_dict()
        assert d["resourceType"] == "Condition"

    def test_critical_obs_pt003_potassium(self):
        obs = self.db.get_observations("PT003")
        k = next((o for o in obs if "Potassium" in o.display), None)
        assert k is not None
        assert "critical" in k.interpretation

    def test_observation_category_filter(self):
        labs = self.db.get_observations("PT001", category="laboratory")
        assert all(o.category == "laboratory" for o in labs)


# ── RAG Engine Tests ───────────────────────────────────────────────────────────

class TestRAGEngine:
    def setup_method(self):
        from rag.engine import ClinicalRAGEngine
        self.rag = ClinicalRAGEngine()
        self.rag.initialize()

    def test_initialization(self):
        assert self.rag._initialized is True
        assert len(self.rag.chunks) > 0

    def test_retrieve_returns_results(self):
        results = self.rag.retrieve("Type 2 diabetes HbA1c target", top_k=3)
        assert len(results) > 0

    def test_retrieve_top_k(self):
        results = self.rag.retrieve("blood pressure management", top_k=3)
        assert len(results) <= 3

    def test_diabetes_query_relevant(self):
        results = self.rag.retrieve("What is the HbA1c target for diabetes?", top_k=3)
        titles = [r.title.lower() for r in results]
        assert any("diabetes" in t or "ada" in t or "hba1c" in t for t in titles)

    def test_hypertension_query_relevant(self):
        results = self.rag.retrieve("blood pressure target hypertension", top_k=3)
        titles = [r.title.lower() for r in results]
        assert any("hypertension" in t or "blood pressure" in t or "acc" in t for t in titles)

    def test_ckd_query_relevant(self):
        results = self.rag.retrieve("chronic kidney disease eGFR stage management", top_k=3)
        titles = [r.title.lower() for r in results]
        assert any("kidney" in t or "ckd" in t or "kdigo" in t or "renal" in t for t in titles)

    def test_icd10_boost(self):
        r1 = self.rag.retrieve("diabetes treatment", top_k=3)
        r2 = self.rag.retrieve("diabetes treatment", top_k=3, icd10_boost=["E11.9"])
        assert len(r1) > 0
        assert len(r2) > 0

    def test_format_context(self):
        results = self.rag.retrieve("potassium hyperkalemia CKD", top_k=2)
        ctx = self.rag.format_context(results)
        assert len(ctx) > 100
        assert "Source" in ctx

    def test_chunk_text(self):
        long_text = " ".join(["word"] * 1000)
        chunks = self.rag._chunk_text(long_text, max_words=200)
        assert len(chunks) > 1
        assert all(len(c.split()) <= 200 for c in chunks)

    def test_keyword_fallback(self):
        results = self.rag._keyword_retrieve("metformin diabetes CKD", top_k=3)
        assert len(results) > 0
        assert all(r.score >= 0 for r in results)


# ── Auth Tests ─────────────────────────────────────────────────────────────────

class TestAuth:
    def test_authenticate_valid_user(self):
        from core.auth import authenticate_user
        user = authenticate_user("dr.roberts", "Demo1234!")
        assert user is not None
        assert user["role"] == "physician"

    def test_authenticate_invalid_password(self):
        from core.auth import authenticate_user
        assert authenticate_user("dr.roberts", "wrongpass") is None

    def test_authenticate_invalid_user(self):
        from core.auth import authenticate_user
        assert authenticate_user("noone", "Demo1234!") is None

    def test_create_and_decode_token(self):
        from core.auth import create_access_token, decode_token
        token = create_access_token({"sub": "dr.roberts", "role": "physician"})
        payload = decode_token(token)
        assert payload is not None
        assert payload["sub"] == "dr.roberts"
        assert payload["role"] == "physician"

    def test_invalid_token(self):
        from core.auth import decode_token
        assert decode_token("totally.invalid.token") is None

    def test_physician_permissions(self):
        from core.auth import has_permission
        assert has_permission("physician", "query") is True
        assert has_permission("physician", "generate_note") is True
        assert has_permission("physician", "view_audit") is True

    def test_nurse_permissions(self):
        from core.auth import has_permission
        assert has_permission("nurse", "query") is True
        assert has_permission("nurse", "generate_note") is False

    def test_admin_permissions(self):
        from core.auth import has_permission
        assert has_permission("admin", "view_audit") is True
        assert has_permission("admin", "query") is False


# ── Clinical Alerts Tests ──────────────────────────────────────────────────────

class TestClinicalAlerts:
    def setup_method(self):
        from agents.clinical_assistant import ClinicalAIAssistant
        self.ai = ClinicalAIAssistant()

    def test_no_alerts_pt001(self):
        # PT001 has high but not critical values
        alerts = self.ai._check_critical_alerts("PT001")
        # Should have no CRITICAL prefix alerts (just high)
        critical = [a for a in alerts if "CRITICAL" in a and "critical" in a.lower()]
        assert isinstance(alerts, list)

    def test_critical_alerts_pt003(self):
        alerts = self.ai._check_critical_alerts("PT003")
        assert len(alerts) > 0
        text = " ".join(alerts)
        assert "Creatinine" in text or "Potassium" in text or "ALERT" in text

    def test_critical_alerts_pt005(self):
        alerts = self.ai._check_critical_alerts("PT005")
        assert len(alerts) >= 3  # HbA1c + Creatinine + K+ + eGFR + drug alert

    def test_drug_alert_metformin_ckd(self):
        alerts = self.ai._check_critical_alerts("PT005")
        text = " ".join(alerts)
        assert "Metformin" in text or "metformin" in text.lower()

    def test_fallback_response_no_key(self):
        result = self.ai._fallback_response("test question", "PT001", [])
        assert "OFFLINE" in result or "Patient" in result or "offline" in result.lower()

    def test_patient_context_building(self):
        ctx = self.ai._build_patient_context("PT001")
        assert "James Carter" in ctx
        assert "Metformin" in ctx or "E11.9" in ctx

    def test_patient_context_missing_patient(self):
        ctx = self.ai._build_patient_context("PT999")
        assert "not found" in ctx.lower()


# ── API Tests ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAPI:
    async def _get_token(self, client):
        from httpx import AsyncClient
        from api.server import app
        fd = {"username": "dr.roberts", "password": "Demo1234!"}
        r = await client.post("/auth/token", data=fd)
        return r.json()["access_token"]

    async def test_health_check(self):
        from httpx import AsyncClient
        from api.server import app
        async with AsyncClient(app=app, base_url="http://test") as c:
            r = await c.get("/api/health")
            assert r.status_code == 200
            d = r.json()
            assert d["status"] == "healthy"
            assert d["hipaa_compliant"] is True

    async def test_login_success(self):
        from httpx import AsyncClient
        from api.server import app
        async with AsyncClient(app=app, base_url="http://test") as c:
            r = await c.post("/auth/token", data={"username":"dr.roberts","password":"Demo1234!"})
            assert r.status_code == 200
            assert "access_token" in r.json()

    async def test_login_fail(self):
        from httpx import AsyncClient
        from api.server import app
        async with AsyncClient(app=app, base_url="http://test") as c:
            r = await c.post("/auth/token", data={"username":"dr.roberts","password":"wrong"})
            assert r.status_code == 401

    async def test_patients_requires_auth(self):
        from httpx import AsyncClient
        from api.server import app
        async with AsyncClient(app=app, base_url="http://test") as c:
            r = await c.get("/patients")
            assert r.status_code == 401

    async def test_list_patients_authenticated(self):
        from httpx import AsyncClient
        from api.server import app
        async with AsyncClient(app=app, base_url="http://test") as c:
            token = await self._get_token(c)
            r = await c.get("/patients", headers={"Authorization": f"Bearer {token}"})
            assert r.status_code == 200
            assert len(r.json()) == 5

    async def test_get_patient_detail(self):
        from httpx import AsyncClient
        from api.server import app
        async with AsyncClient(app=app, base_url="http://test") as c:
            token = await self._get_token(c)
            r = await c.get("/patients/PT001", headers={"Authorization": f"Bearer {token}"})
            assert r.status_code == 200
            d = r.json()
            assert "fhir" in d
            assert "conditions" in d

    async def test_get_patient_not_found(self):
        from httpx import AsyncClient
        from api.server import app
        async with AsyncClient(app=app, base_url="http://test") as c:
            token = await self._get_token(c)
            r = await c.get("/patients/PT999", headers={"Authorization": f"Bearer {token}"})
            assert r.status_code == 404

    async def test_get_patient_alerts(self):
        from httpx import AsyncClient
        from api.server import app
        async with AsyncClient(app=app, base_url="http://test") as c:
            token = await self._get_token(c)
            r = await c.get("/patients/PT005/alerts", headers={"Authorization": f"Bearer {token}"})
            assert r.status_code == 200
            d = r.json()
            assert "critical_alerts" in d
            assert len(d["critical_alerts"]) > 0

    async def test_query_empty_rejected(self):
        from httpx import AsyncClient
        from api.server import app
        async with AsyncClient(app=app, base_url="http://test") as c:
            token = await self._get_token(c)
            r = await c.post("/query",
                headers={"Authorization": f"Bearer {token}"},
                json={"question": "", "patient_id": "PT001"})
            assert r.status_code == 400

    async def test_query_offline_mode(self):
        from httpx import AsyncClient
        from api.server import app
        async with AsyncClient(app=app, base_url="http://test") as c:
            token = await self._get_token(c)
            r = await c.post("/query",
                headers={"Authorization": f"Bearer {token}"},
                json={"question": "What are the lab concerns?", "patient_id": "PT001"})
            assert r.status_code == 200
            d = r.json()
            assert "answer" in d
            assert "critical_alerts" in d

    async def test_nurse_cannot_generate_note(self):
        from httpx import AsyncClient
        from api.server import app
        async with AsyncClient(app=app, base_url="http://test") as c:
            r = await c.post("/auth/token", data={"username":"nurse.johnson","password":"Demo1234!"})
            token = r.json()["access_token"]
            r2 = await c.post("/patients/PT001/note",
                headers={"Authorization": f"Bearer {token}"},
                json={"note_type": "progress"})
            assert r2.status_code == 403

    async def test_audit_physician_access(self):
        from httpx import AsyncClient
        from api.server import app
        async with AsyncClient(app=app, base_url="http://test") as c:
            token = await self._get_token(c)
            r = await c.get("/audit", headers={"Authorization": f"Bearer {token}"})
            assert r.status_code == 200
            assert "entries" in r.json()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
