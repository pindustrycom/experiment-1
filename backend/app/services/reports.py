"""Research report (PDF) generator."""
from __future__ import annotations

import io
from datetime import UTC, datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.serializers import pair_out
from app.core.config import get_settings
from app.models import Pair
from app.services import performance_service


def _fmt(v, nd=3):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _table(rows, widths=None):
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2933")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9aa5b1")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def build_pdf(db: Session) -> bytes:
    s = get_settings()
    st = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm, bottomMargin=15 * mm,
                            title="Splittability Monitor research report")
    perf = performance_service.performance_report(db, n_boot=300)
    el = [Paragraph("Splittability Monitor — research report", st["Title"]),
          Paragraph(f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M UTC}. Mode: <b>{s.app_mode}</b>.", st["Normal"])]
    if s.is_demo or perf["live"]["contains_simulated"]:
        el.append(Paragraph("<font color='#b45309'><b>Contains SIMULATED demo data. Not research evidence.</b></font>", st["Normal"]))
    el += [Paragraph("Scores are relative risk indices, not calibrated probabilities. Live forward results and historical backtests are reported separately; unresolved predictions are excluded.", st["Italic"]), Spacer(1, 6)]

    el.append(Paragraph("Current watchlist", st["Heading2"]))
    pairs = pair_out(db, db.scalars(select(Pair).where(Pair.status == "active").order_by(Pair.slug)).all())
    rows = [["Pair", "SI-v1", "SI-v2", "Status", "7d Δ", "Bias", "Confidence"]]
    for p in pairs:
        rows.append([p.slug, _fmt(p.si_v1.value if p.si_v1 else None, 1), _fmt(p.si_v2.value if p.si_v2 else None, 1),
                     p.si_v2.status if p.si_v2 else "–", _fmt(p.si_v2.change_7d if p.si_v2 else None, 1), p.direction or "–", _fmt(p.confidence, 2)])
    el.append(_table(rows))

    el.append(Paragraph("Live forward test", st["Heading2"]))
    for h, hr in perf["live"]["horizons"].items():
        el.append(Paragraph(f"Horizon {h} sessions — threshold {hr['target']['threshold']:.3f}; resolved {hr['resolved']}, pending {hr['pending']}", st["Heading4"]))
        rows = [["Index", "n", "ROC-AUC", "95% CI (industry bootstrap)", "PR-AUC", "Brier*", "Status"]]
        for name, m in hr["indices"].items():
            ci = m.get("roc_auc_ci")
            rows.append([name, m.get("n"), _fmt(m.get("roc_auc")), f"{ci['low']:.3f}–{ci['high']:.3f}" if ci else "–", _fmt(m.get("pr_auc")),
                         _fmt(m.get("brier")), m.get("status") if m.get("status") == "ok" else m.get("reason", "")[:60]])
        el.append(_table(rows))
    el.append(Paragraph("* Brier uses score/100 as a pseudo-probability.", st["Italic"]))
    el.append(PageBreak())
    el.append(Paragraph("Backtests and reported benchmark", st["Heading2"]))
    for b in perf["backtests"]:
        el.append(Paragraph(f"{b['label']} ({b['kind']}{', SIMULATED' if b['is_demo'] else ''})", st["Heading4"]))
        if b["kind"] == "reported_historical":
            rows = [["Index", "ROC-AUC"]] + [[k, _fmt(v.get("roc_auc"), 4)] for k, v in b["metrics"].items()]
        else:
            rows = [["Index (test period)", "n", "ROC-AUC", "PR-AUC", "Status"]]
            for k, m in b["metrics"].get("test", {}).items():
                rows.append([k, m.get("n"), _fmt(m.get("roc_auc")), _fmt(m.get("pr_auc")), m.get("status")])
        el.append(_table(rows))
        for lim in b["limitations"]:
            el.append(Paragraph(f"• {lim}", st["Normal"]))
    doc.build(el)
    return buf.getvalue()
