from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import current_user, require_admin
from app.core.db import get_db
from app.models import Pair, User, Watchlist, WatchlistPair
from app.schemas.api import WatchlistIn, WatchlistOut

router = APIRouter(prefix="/watchlists", tags=["watchlists"])


def _out(w: Watchlist) -> WatchlistOut:
    return WatchlistOut(id=w.id, name=w.name, description=w.description, is_default=w.is_default, pair_ids=[p.id for p in w.pairs])


@router.get("", response_model=list[WatchlistOut])
def list_watchlists(db: Session = Depends(get_db), _: User = Depends(current_user)):
    return [_out(w) for w in db.scalars(select(Watchlist).order_by(Watchlist.name)).all()]


@router.post("", response_model=WatchlistOut, status_code=201)
def create_watchlist(body: WatchlistIn, db: Session = Depends(get_db), _: User = Depends(require_admin)):
    if db.scalar(select(Watchlist).where(Watchlist.name == body.name)):
        raise HTTPException(409, "watchlist name already used")
    w = Watchlist(name=body.name, description=body.description)
    db.add(w)
    db.flush()
    for pid in set(body.pair_ids):
        if db.get(Pair, pid):
            db.add(WatchlistPair(watchlist_id=w.id, pair_id=pid))
    db.commit()
    db.refresh(w)
    return _out(w)


@router.put("/{wid}", response_model=WatchlistOut)
def update_watchlist(wid: int, body: WatchlistIn, db: Session = Depends(get_db), _: User = Depends(require_admin)):
    w = db.get(Watchlist, wid)
    if w is None:
        raise HTTPException(404, "watchlist not found")
    w.name, w.description = body.name, body.description
    db.query(WatchlistPair).filter(WatchlistPair.watchlist_id == wid).delete()
    for pid in set(body.pair_ids):
        if db.get(Pair, pid):
            db.add(WatchlistPair(watchlist_id=wid, pair_id=pid))
    db.commit()
    db.refresh(w)
    return _out(w)


@router.delete("/{wid}", status_code=204)
def delete_watchlist(wid: int, db: Session = Depends(get_db), _: User = Depends(require_admin)):
    """Deleting a watchlist only removes the grouping; pairs and their history are untouched."""
    w = db.get(Watchlist, wid)
    if w is None:
        raise HTTPException(404, "watchlist not found")
    db.delete(w)
    db.commit()
