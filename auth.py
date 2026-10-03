"""
JWT authentication using the vBiz USER_LOGIN table.
OFBiz stores passwords as SHA-1 hex (uppercase).
"""
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config import JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRE_MINS

security = HTTPBearer(auto_error=False)


def _sha1_hex(password: str) -> str:
    return hashlib.sha1(password.encode("utf-8")).hexdigest().upper()


def verify_user(user_login_id: str, password: str, db_conn) -> Optional[dict]:
    """
    Check credentials against USER_LOGIN table.
    Returns user dict or None if invalid.
    """
    try:
        cursor = db_conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT USER_LOGIN_ID, CURRENT_PASSWORD, IS_SYSTEM, PARTY_ID "
            "FROM USER_LOGIN WHERE USER_LOGIN_ID = %s AND ENABLED = 'Y'",
            (user_login_id,)
        )
        row = cursor.fetchone()
        cursor.close()
        if not row:
            return None

        stored = (row.get("CURRENT_PASSWORD") or "").strip()
        # OFBiz SHA-1 hash comparison
        if stored and stored != _sha1_hex(password):
            return None

        return {
            "user_login_id": row["USER_LOGIN_ID"],
            "party_id":      row.get("PARTY_ID", ""),
            "is_system":     row.get("IS_SYSTEM") == "Y",
        }
    except Exception:
        return None


def create_token(user_info: dict) -> str:
    payload = {
        **user_info,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    return decode_token(credentials.credentials)


def get_optional_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> Optional[dict]:
    if not credentials:
        return None
    try:
        return decode_token(credentials.credentials)
    except HTTPException:
        return None
