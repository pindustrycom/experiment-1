"""Operational commands: python -m app.cli <command>"""
from __future__ import annotations

import argparse
import getpass
import json
import sys

from sqlalchemy import func, select

from app.core.config import get_settings
from app.core.db import session_scope
from app.core.logging import configure_logging
from app.core.security import hash_password, validate_password_strength
from app.models import PredictionSnapshot, PriceObservation, User


def create_admin(email: str, password: str | None) -> None:
    password = password or getpass.getpass("Password: ")
    if (msg := validate_password_strength(password)):
        sys.exit(msg)
    with session_scope() as db:
        u = db.scalar(select(User).where(User.email == email.lower()))
        if u:
            u.password_hash, u.role, u.is_active = hash_password(password), "admin", True
            print(f"updated admin {email}")
        else:
            db.add(User(email=email.lower(), password_hash=hash_password(password), role="admin"))
            print(f"created admin {email}")


def init(seed_demo: bool) -> None:
    """Idempotent bootstrap: reference data, model versions, schedules, optional demo data and the
    bootstrap admin from BOOTSTRAP_ADMIN_EMAIL/PASSWORD."""
    from app.services import backtest_service, demo_service, jobs
    from app.services.reference import seed_reference_data
    s = get_settings()
    with session_scope() as db:
        print(json.dumps(seed_reference_data(db)))
        backtest_service.seed_reported_benchmark(db)
        jobs.ensure_scheduled_jobs(db, s.is_demo)
    if s.bootstrap_admin_email and s.secret("bootstrap_admin_password"):
        with session_scope() as db:
            if not db.scalar(select(User).where(User.email == s.bootstrap_admin_email.lower())):
                db.add(User(email=s.bootstrap_admin_email.lower(), password_hash=hash_password(s.secret("bootstrap_admin_password")), role="admin"))
                print(f"bootstrap admin {s.bootstrap_admin_email} created")
    if s.is_demo and seed_demo:
        with session_scope() as db:
            if not demo_service.is_seeded(db):
                print("seeding SIMULATED demo data (a few minutes)...")
                print(json.dumps({k: v for k, v in demo_service.seed(db).items() if k != "scores"}, default=str))


def wipe_demo(confirm: bool) -> None:
    from app.services import demo_service
    with session_scope() as db:
        real = db.scalar(select(func.count()).select_from(PriceObservation).where(PriceObservation.is_demo.is_(False)))
        real += db.scalar(select(func.count()).select_from(PredictionSnapshot).where(PredictionSnapshot.is_demo.is_(False)))
        if real:
            sys.exit("refusing: database contains non-demo research data")
        if not confirm:
            sys.exit("pass --yes to delete all simulated data")
        demo_service.wipe(db)
        print("demo data removed")


def run_job(name: str, params: str) -> None:
    from app.services import jobs, usage
    usage.install()
    r = jobs.run_now(name, trigger="manual", params=json.loads(params), demo=get_settings().is_demo)
    print(json.dumps({"status": r.status if r else "skipped", "error": r.error if r else None, "result": r.result if r else None}, default=str, indent=2)[:20000])


def main() -> None:
    configure_logging(get_settings().log_level)
    ap = argparse.ArgumentParser(prog="python -m app.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("create-admin")
    a.add_argument("--email", required=True)
    a.add_argument("--password", help="omit to be prompted")
    i = sub.add_parser("init")
    i.add_argument("--no-demo", action="store_true")
    w = sub.add_parser("wipe-demo")
    w.add_argument("--yes", action="store_true")
    j = sub.add_parser("run-job")
    j.add_argument("name")
    j.add_argument("--params", default="{}")
    args = ap.parse_args()
    if args.cmd == "create-admin":
        create_admin(args.email, args.password)
    elif args.cmd == "init":
        init(not args.no_demo)
    elif args.cmd == "wipe-demo":
        wipe_demo(args.yes)
    elif args.cmd == "run-job":
        run_job(args.name, args.params)


if __name__ == "__main__":
    main()
