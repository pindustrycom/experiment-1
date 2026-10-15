from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from app.core.config import get_settings

_hasher = PasswordHasher()  # Argon2id with library-recommended parameters

SESSION_COOKIE = "sm_session"
JWT_ALG = "HS256"


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(password_hash: str) -> bool:
    return _hasher.check_needs_rehash(password_hash)


def create_session_token(user_id: int, role: str) -> str:
    s = get_settings()
    now = datetime.now(UTC)
    payload = {"sub": str(user_id), "role": role, "iat": now, "exp": now + timedelta(hours=s.session_ttl_hours)}
    return jwt.encode(payload, s.secret("secret_key"), algorithm=JWT_ALG)


def decode_session_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, get_settings().secret("secret_key"), algorithms=[JWT_ALG])
    except jwt.PyJWTError:
        return None


def validate_password_strength(password: str) -> str | None:
    if len(password) < 12:
        return "Password must be at least 12 characters"
    if password.lower() == password or password.upper() == password:
        return "Password must mix upper and lower case"
    if not any(c.isdigit() for c in password):
        return "Password must contain a digit"
    return None
