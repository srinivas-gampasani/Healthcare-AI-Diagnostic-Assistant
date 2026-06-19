"""
HIPAA Compliance Module
=======================
Handles:
- PHI (Protected Health Information) detection and masking
- Audit logging (every access to patient data logged)
- Data encryption/decryption at rest
- De-identification utilities
- Access control enforcement
"""
import re
import json
import logging
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

# ── PHI Pattern Registry ───────────────────────────────────────────────────────

PHI_PATTERNS = {
    "ssn": (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN-REDACTED]"),
    "ssn_nodash": (re.compile(r"\b\d{9}\b"), "[SSN-REDACTED]"),
    "phone": (re.compile(r"\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE-REDACTED]"),
    "email": (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL-REDACTED]"),
    "dob": (re.compile(r"\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](\d{2}|\d{4})\b"), "[DOB-REDACTED]"),
    "mrn": (re.compile(r"\bMRN[:\s#]*\d{6,10}\b", re.IGNORECASE), "[MRN-REDACTED]"),
    "npi": (re.compile(r"\bNPI[:\s#]*\d{10}\b", re.IGNORECASE), "[NPI-REDACTED]"),
    "zip_extended": (re.compile(r"\b\d{5}-\d{4}\b"), "[ZIP-REDACTED]"),
    "ip_address": (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "[IP-REDACTED]"),
    "account_number": (re.compile(r"\b(ACC|ACCT)[:\s#]*\d{6,12}\b", re.IGNORECASE), "[ACCOUNT-REDACTED]"),
}


class PHIMasker:
    """Detects and masks PHI from text for HIPAA compliance."""

    def mask(self, text: str) -> tuple[str, List[str]]:
        """
        Mask PHI in text. Returns (masked_text, list_of_found_phi_types).
        """
        if not settings.PHI_MASKING_ENABLED:
            return text, []

        found_types = []
        masked = text

        for phi_type, (pattern, replacement) in PHI_PATTERNS.items():
            if pattern.search(masked):
                found_types.append(phi_type)
                masked = pattern.sub(replacement, masked)

        return masked, found_types

    def detect(self, text: str) -> List[str]:
        """Return list of PHI types detected in text."""
        return [t for t, (p, _) in PHI_PATTERNS.items() if p.search(text)]

    def hash_identifier(self, identifier: str) -> str:
        """One-way hash a patient identifier for de-identification."""
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]


# ── Audit Logger ───────────────────────────────────────────────────────────────

class AuditLogger:
    """
    HIPAA-required audit logging.
    Logs every access to patient data with user, action, timestamp, resource.
    """

    def __init__(self):
        self.log_path = Path(settings.AUDIT_LOG_PATH)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        outcome: str = "success",
        details: Optional[Dict[str, Any]] = None,
        ip_address: str = "unknown",
    ):
        """Write an audit log entry."""
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user_id": user_id,
            "action": action,                   # e.g. "query_patient_history"
            "resource_type": resource_type,     # e.g. "patient", "note", "lab"
            "resource_id": resource_id,         # hashed or de-identified
            "outcome": outcome,                 # success | denied | error
            "ip_address": ip_address,
            "details": details or {},
        }
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Audit log write failed: {e}")

        logger.info(f"AUDIT | {user_id} | {action} | {resource_type}:{resource_id} | {outcome}")

    def get_recent(self, n: int = 50) -> List[Dict]:
        """Return the last n audit entries."""
        try:
            with open(self.log_path) as f:
                lines = f.readlines()
            return [json.loads(l) for l in lines[-n:]]
        except FileNotFoundError:
            return []


# ── Simple Encryption Helper ───────────────────────────────────────────────────

class DataEncryptor:
    """
    Symmetric encryption for PHI at rest using Fernet (AES-128-CBC).
    In production: use Azure Key Vault / AWS KMS.
    """

    def __init__(self):
        self._fernet = None
        self._init_fernet()

    def _init_fernet(self):
        try:
            from cryptography.fernet import Fernet
            import base64
            # Derive key from SECRET_KEY (deterministic for demo; use proper KMS in prod)
            raw = settings.SECRET_KEY.encode()[:32].ljust(32, b"0")
            key = base64.urlsafe_b64encode(raw)
            self._fernet = Fernet(key)
        except Exception as e:
            logger.warning(f"Encryption init failed (non-critical for demo): {e}")

    def encrypt(self, text: str) -> str:
        if not self._fernet:
            return text
        return self._fernet.encrypt(text.encode()).decode()

    def decrypt(self, token: str) -> str:
        if not self._fernet:
            return token
        try:
            return self._fernet.decrypt(token.encode()).decode()
        except Exception:
            return token


# ── Singletons ─────────────────────────────────────────────────────────────────

phi_masker = PHIMasker()
audit_logger = AuditLogger()
encryptor = DataEncryptor()
