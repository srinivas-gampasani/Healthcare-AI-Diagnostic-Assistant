"""
Authentication & Authorization
===============================
JWT-based auth with clinical role-based access control (RBAC).

Roles:
- physician: full access including note generation
- nurse: patient query, vitals, medications (read)
- admin: audit logs, system management
- care_coordinator: limited patient data access
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from jose import JWTError, jwt
from config import settings

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ALGORITHM = "HS256"

# Role permissions
ROLE_PERMISSIONS: Dict[str, list] = {
    "physician": ["query", "generate_note", "view_patient", "view_labs", "view_meds", "view_audit"],
    "nurse": ["query", "view_patient", "view_labs", "view_meds"],
    "care_coordinator": ["query", "view_patient", "view_labs"],
    "admin": ["view_audit", "system_admin"],
}

# Demo users (in production: database-backed)
DEMO_USERS = {
    "dr.roberts": {
        "username": "dr.roberts",
        "full_name": "Dr. Emily Roberts",
        "email": "e.roberts@ascension.org",
        "role": "physician",
        "department": "Internal Medicine",
        "hashed_password": pwd_context.hash("Demo1234!"),
        "active": True,
    },
    "nurse.johnson": {
        "username": "nurse.johnson",
        "full_name": "Nurse Sarah Johnson",
        "email": "s.johnson@ascension.org",
        "role": "nurse",
        "department": "Cardiology",
        "hashed_password": pwd_context.hash("Demo1234!"),
        "active": True,
    },
    "admin": {
        "username": "admin",
        "full_name": "System Administrator",
        "email": "admin@ascension.org",
        "role": "admin",
        "department": "IT",
        "hashed_password": pwd_context.hash("Admin1234!"),
        "active": True,
    },
}


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def get_user(username: str) -> Optional[Dict]:
    return DEMO_USERS.get(username)


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    user = get_user(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode["exp"] = expire
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[Dict]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def has_permission(role: str, permission: str) -> bool:
    return permission in ROLE_PERMISSIONS.get(role, [])
