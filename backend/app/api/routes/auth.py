from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import current_user, require_admin
from app.api.ratelimit import client_ip, limiter
from app.core.config import get_settings
from app.core.db import get_db
from app.core.security import SESSION_COOKIE, create_session_token, hash_password, needs_rehash, validate_password_strength, verify_password
from app.models import User
from app.schemas.api import ChangePassword, LoginRequest, UserCreate, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])
_DUMMY_HASH = hash_password("timing-equaliser-not-a-real-password")


@router.post("/login", response_model=UserOut)
def login(body: LoginRequest, request: Request, response: Response, db: Session = Depends(get_db)):
    s = get_settings()
    ok, retry = limiter.allow(f"login:{client_ip(request)}", s.login_rate_limit_per_minute)
    if not ok:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, f"too many login attempts; retry in {retry}s")
    user = db.scalar(select(User).where(User.email == body.email.strip().lower()))
    valid = verify_password(user.password_hash if user else _DUMMY_HASH, body.password)
    if not user or not valid or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid email or password")
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(body.password)
    user.last_login_at = datetime.now(UTC)
    db.commit()
    response.set_cookie(SESSION_COOKIE, create_session_token(user.id, user.role), httponly=True, secure=s.cookie_secure,
                        samesite="strict", max_age=s.session_ttl_hours * 3600, path="/")
    return user


@router.post("/logout", status_code=204)
def logout(response: Response):
    response.delete_cookie(SESSION_COOKIE, path="/")


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    return user


@router.post("/change-password", status_code=204)
def change_password(body: ChangePassword, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if not verify_password(user.password_hash, body.current_password):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "current password is incorrect")
    if (msg := validate_password_strength(body.new_password)):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, msg)
    db.get(User, user.id).password_hash = hash_password(body.new_password)
    db.commit()


@router.get("/users", response_model=list[UserOut])
def list_users(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    return db.scalars(select(User).order_by(User.id)).all()


@router.post("/users", response_model=UserOut, status_code=201)
def create_user(body: UserCreate, _: User = Depends(require_admin), db: Session = Depends(get_db)):
    if (msg := validate_password_strength(body.password)):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, msg)
    email = body.email.lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "user already exists")
    u = User(email=email, password_hash=hash_password(body.password), role=body.role)
    db.add(u)
    db.commit()
    return u
