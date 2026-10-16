"""Unit tests for SI-v1, SI-v2 components, missing-data handling and look-ahead safeguards."""
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from app.scoring.components import effective_asymmetry, event_side_sign, score_news, sic_distance
from app.scoring.params import SI_V2_PARAMS, normalize_weights, si_v2_definition
from app.scoring.si_v1 import compute_si_v1, index_components
from app.scoring.si_v2 import combine, compute_si_v2
from app.scoring.types import ComponentResult
from tests.factories import correlated_prices, coverage, event, facts, market, pair_inputs

PN = SI_V2_PARAMS["N"]
CUT = datetime(2026, 9, 30, 22, tzinfo=UTC)


class TestSIv1:
    def test_formula_and_range(self):
        r = compute_si_v1(market(correlated_prices(n=1400, rho=0.8)))
        c, d, rr = (r.components[k].value for k in "CDR")
        assert r.value == pytest.approx((25 * c + 15 * d + 25 * rr) * 100 / 65, abs=1e-3)
        assert 0 <= r.value <= 100

    @pytest.mark.parametrize("args,expected", [((0.9, 0.9, 0), 0.0), ((-1.0, 0.9, 80), 100.0), ((-0.1, 0.9, 60), 100.0)])
    def test_matches_original_index_components(self, args, expected):
        # Reference values from the original run_experiment.py self_test().
        c, d, r = index_components(*args)
        assert (25 * c + 15 * d + 25 * r) * 100 / 65 == pytest.approx(expected, abs=1e-10)

    def test_matches_original_calculate_rows(self):
        prices = correlated_prices(n=1500, rho=0.7, seed=11, drift_a=0.0003)
        r = compute_si_v1(market(prices))
        past = prices.iloc[-1261:]
        rets = np.log(past).diff().dropna()
        rho5, rho52 = rets.iloc[-1260:].corr().iloc[0, 1], rets.iloc[-252:].corr().iloc[0, 1]
        gap = abs(100 * (past.a.iloc[-1] / past.a.iloc[-253] - 1) - 100 * (past.b.iloc[-1] / past.b.iloc[-253] - 1))
        c = 25 * min(1, max(0, (0.9 - rho52) / 0.9))
        d = 15 * min(1, max(0, (rho5 - rho52) / 0.3))
        rr = 25 * min(1, max(0, gap / 50))
        assert r.value == pytest.approx((c + d + rr) * 100 / 65, abs=1e-3)
        assert r.baselines["rho_short"] == pytest.approx(rho52)
        assert r.baselines["rho_long"] == pytest.approx(rho5)

    def test_high_correlation_scores_lower_than_broken_relationship(self):
        stable = compute_si_v1(market(correlated_prices(n=1400, rho=0.9, seed=5)))
        broken = compute_si_v1(market(correlated_prices(n=1400, rho=0.9, seed=5, break_at=1250, rho_after=0.0, drift_a=0.0008)))
        assert broken.value > stable.value
        assert broken.components["D"].value > stable.components["D"].value

    def test_insufficient_history_is_unavailable_not_zero(self):
        r = compute_si_v1(market(correlated_prices(n=1000)))
        assert r.value is None
        assert not r.components["C"].available
        assert r.baselines["correlation_only"] is None

    def test_baselines(self):
        r = compute_si_v1(market(correlated_prices(n=1400)))
        assert r.baselines["correlation_only"] == pytest.approx(100 * (1 - r.baselines["rho_short"]))
        assert r.baselines["previous_return_gap"] == pytest.approx(100 * r.components["R"].value)


class TestWeights:
    def test_points_and_fractions_equivalent(self):
        assert normalize_weights({"N": 35, "F": 25, "M": 20, "P": 10, "S": 10}) == normalize_weights(
            {"N": 0.35, "F": 0.25, "M": 0.2, "P": 0.1, "S": 0.1})

    def test_version_changes_with_weights(self):
        a = si_v2_definition()
        b = si_v2_definition({"N": 30, "F": 30, "M": 20, "P": 10, "S": 10})
        assert a["version"] != b["version"]
        assert si_v2_definition()["version"] == a["version"]  # deterministic

    @pytest.mark.parametrize("bad", [{"N": -1, "F": 1, "M": 1, "P": 1, "S": 1}, {"N": 0, "F": 0, "M": 0, "P": 0, "S": 0}, {"N": 1}])
    def test_invalid_weights(self, bad):
        with pytest.raises(ValueError):
            normalize_weights(bad)


class TestCombine:
    W = {"N": 0.35, "F": 0.25, "M": 0.2, "P": 0.1, "S": 0.1}

    def comps(self, **vals):
        return {k: ComponentResult(k, vals.get(k), vals.get(k) is not None, "") for k in "NFMPS"}

    def test_complete(self):
        v, s, cov = combine(self.comps(N=1, F=0, M=0.5, P=0.5, S=0), self.W, 0.6, ["M"])
        assert s == "complete" and cov == pytest.approx(1)
        assert v == pytest.approx(35 + 10 + 5)

    def test_provisional_renormalises(self):
        v, s, cov = combine(self.comps(N=None, F=0.5, M=0.5, P=None, S=0.5), self.W, 0.5, ["M"])
        assert s == "provisional" and cov == pytest.approx(0.55)
        assert v == pytest.approx(50)

    def test_below_min_coverage_unavailable(self):
        v, s, _ = combine(self.comps(M=0.9, S=0.9), self.W, 0.6, ["M"])
        assert v is None and s == "unavailable"

    def test_required_component_missing(self):
        v, s, _ = combine(self.comps(N=0.5, F=0.5, P=0.5, S=0.5), self.W, 0.6, ["M"])
        assert v is None and s == "unavailable"


class TestNews:
    def test_symmetric_industry_event_contributes_little(self):
        common = event(direction="symmetric", scope="industry_wide", asym=0.9, severity=0.8, etype="industry_shock")
        specific = event(i=2, direction="PEP_disadvantaged", scope="company_specific", asym=0.9, severity=0.8)
        n_common = score_news([common], coverage(), CUT, "KO", "PEP", PN)
        n_spec = score_news([specific], coverage(), CUT, "KO", "PEP", PN)
        assert effective_asymmetry(common, PN["shared_scope_asymmetry_cap"]) <= 0.3
        assert n_common.value < n_spec.value / 2

    def test_repeated_coverage_counts_once(self):
        one = score_news([event(i=1, thread="recall")], coverage(), CUT, "KO", "PEP", PN)
        many = score_news([event(i=k, thread="recall") for k in range(1, 8)], coverage(), CUT, "KO", "PEP", PN)
        assert many.value == pytest.approx(one.value)

    def test_unverified_evidence_ignored(self):
        r = score_news([event(verified=0.0)], coverage(), CUT, "KO", "PEP", PN)
        assert r.available and r.value == 0

    def test_not_material_ignored(self):
        assert score_news([event(material=False)], coverage(), CUT, "KO", "PEP", PN).value == 0

    def test_future_published_event_excluded(self):
        fut = event(published=CUT + timedelta(days=1))
        assert score_news([fut], coverage(), CUT, "KO", "PEP", PN).value == 0

    def test_late_ingested_event_excluded(self):
        """Published before the cutoff but only ingested afterwards: not knowable at the cutoff."""
        late = event(published=CUT - timedelta(days=2), ingested=CUT + timedelta(hours=3))
        assert score_news([late], coverage(), CUT, "KO", "PEP", PN).value == 0

    def test_decay_with_age(self):
        recent = score_news([event(published=CUT - timedelta(days=2))], coverage(), CUT, "KO", "PEP", PN)
        old = score_news([event(published=CUT - timedelta(days=120))], coverage(), CUT, "KO", "PEP", PN)
        assert recent.value > old.value > 0

    def test_no_coverage_unavailable(self):
        assert not score_news([], None, CUT, "KO", "PEP", PN).available
        assert not score_news([], coverage(ok=1), CUT, "KO", "PEP", PN).available
        assert not score_news([], coverage(relevant=10, analyzed=2), CUT, "KO", "PEP", PN).available

    def test_no_events_with_coverage_is_measured_zero(self):
        r = score_news([], coverage(), CUT, "KO", "PEP", PN)
        assert r.available and r.value == 0

    def test_direction_sign(self):
        assert event_side_sign(event(direction="PEP_disadvantaged"), "KO", "PEP") == 1
        assert event_side_sign(event(direction="KO_disadvantaged"), "KO", "PEP") == -1
        assert event_side_sign(event(direction="PEP_advantaged"), "KO", "PEP") == -1
        assert event_side_sign(event(direction="unclear"), "KO", "PEP") == 0


class TestSIv2:
    def test_full_score_with_all_components(self):
        fa = facts("KO", 12.5e9, 11.8e9, 3.6e9, 3.3e9, 0.80, 0.72)
        fb = facts("PEP", 22e9, 22.5e9, 3.2e9, 3.6e9, 1.9, 2.1)
        r = compute_si_v2(pair_inputs(events=[event()], fa=fa, fb=fb))
        assert r.status == "complete" and r.coverage == pytest.approx(1)
        assert 0 <= r.value <= 100
        assert r.direction == "a_advantaged"  # PEP disadvantaged + weaker PEP fundamentals
        assert 0 <= r.confidence <= 1
        assert r.supporting_event_ids == [1]

    def test_no_news_pipeline_gives_provisional_not_zero_risk(self):
        fa = facts("KO", 12.5e9, 11.8e9, 3.6e9, 3.3e9, 0.80, 0.72)
        fb = facts("PEP", 22e9, 22.5e9, 3.2e9, 3.6e9, 1.9, 2.1)
        r = compute_si_v2(pair_inputs(events=None, cov=None, fa=fa, fb=fb))
        assert not r.components["N"].available and not r.components["P"].available
        # F+M+S = 55% of weight < default 60% minimum: the news index refuses to score without news
        assert r.status == "unavailable" and r.value is None and r.coverage == pytest.approx(0.55)
        assert "Asymmetric news risk" in r.explanation
        from copy import deepcopy
        params = deepcopy(SI_V2_PARAMS)
        params["min_coverage"] = 0.5
        r2 = compute_si_v2(pair_inputs(events=None, cov=None, fa=fa, fb=fb), params=params)
        assert r2.status == "provisional" and r2.value is not None

    def test_market_only_is_unavailable(self):
        r = compute_si_v2(pair_inputs(events=None, cov=None, sic=(None, None)))
        assert r.value is None and r.status == "unavailable"
        assert r.direction == "uncertain"

    def test_future_fundamentals_not_used(self):
        """Facts filed after the cutoff must not influence the score."""
        fa = facts("KO", 12.5e9, 11.8e9, 3.6e9, 3.3e9, 0.80, 0.72, filed=datetime(2026, 10, 20).date())
        fb = facts("PEP", 22e9, 22.5e9, 3.2e9, 3.6e9, 1.9, 2.1, filed=datetime(2026, 10, 20).date())
        r = compute_si_v2(pair_inputs(events=[event()], fa=fa, fb=fb))
        assert not r.components["F"].available

    def test_price_rows_after_cutoff_do_not_change_score(self):
        """Scores must depend only on data up to the cutoff."""
        prices = correlated_prices(n=800)
        cut_idx = prices.index[700]
        cutoff = datetime(cut_idx.year, cut_idx.month, cut_idx.day, 22, tzinfo=UTC)
        from app.scoring.pit import truncate_market
        a = compute_si_v2(pair_inputs(prices=truncate_market(prices, cutoff), events=[], cutoff=cutoff))
        mutated = prices.copy()
        mutated.iloc[701:, 0] *= 3  # wild future moves
        b = compute_si_v2(pair_inputs(prices=truncate_market(mutated, cutoff), events=[], cutoff=cutoff))
        assert a.value == b.value

    def test_conflicting_evidence_uncertain(self):
        fa = facts("KO", 10e9, 12e9, 2e9, 3e9, 0.5, 0.9)  # KO fundamentals weaken
        fb = facts("PEP", 24e9, 20e9, 4e9, 3e9, 2.5, 1.8)
        ev = [event(i=1, severity=1, asym=1, persistence=365, thread="a"), event(i=2, severity=1, asym=1, persistence=365, thread="b")]
        r = compute_si_v2(pair_inputs(events=ev, fa=fa, fb=fb))
        assert r.direction == "uncertain"

    def test_sic_distance(self):
        assert sic_distance("2086", "2086") == 0
        assert sic_distance("2086", "2080") == pytest.approx(1 / 3)
        assert sic_distance("2086", "2099") == pytest.approx(2 / 3)
        assert sic_distance("2086", "6021") == 1
        assert sic_distance(None, "1") is None


def test_restatement_filed_after_cutoff_is_not_used():
    from datetime import date

    from app.scoring.components import fundamental_metrics
    from app.scoring.types import CompanyProfile, Fact
    facts = [
        Fact("revenue", date(2025, 6, 30), 91, 100.0, date(2025, 7, 25), "a0"),
        Fact("revenue", date(2026, 6, 30), 91, 110.0, date(2026, 7, 25), "a1"),
        Fact("revenue", date(2026, 6, 30), 91, 130.0, date(2026, 10, 1), "a1-restated"),  # restatement filed later
    ]
    prof = CompanyProfile(ticker="KO", sic_code=None, facts=facts)
    before = fundamental_metrics(prof, date(2026, 9, 1), {"max_staleness_days": 200})
    after = fundamental_metrics(prof, date(2026, 10, 2), {"max_staleness_days": 200})
    assert before["revenue_growth"] == pytest.approx(0.10)
    assert after["revenue_growth"] == pytest.approx(0.30)


def test_fundamentals_compare_quarters_of_slightly_different_length():
    """Coca-Cola's Q1 2026 ran 92 days and Q1 2025 ran 87 (52/53-week calendar); they are still the same quarter."""
    from datetime import date

    from app.scoring.components import fundamental_metrics
    from app.scoring.types import CompanyProfile, Fact
    rows = []
    for m, cur, prev in [("revenue", 12.5e9, 11.1e9), ("operating_income", 4.0e9, 3.6e9), ("eps_diluted", 0.85, 0.77)]:
        rows += [Fact(m, date(2026, 4, 3), 92, cur, date(2026, 4, 30), "a1", "10-Q"),
                 Fact(m, date(2025, 3, 28), 87, prev, date(2025, 4, 29), "a0", "10-Q"),
                 Fact(m, date(2025, 12, 31), 364, prev * 4, date(2026, 2, 20), "k1", "10-K")]
    out = fundamental_metrics(CompanyProfile("KO", facts=rows), date(2026, 10, 15), SI_V2_PARAMS["F"])
    assert {"revenue_growth", "op_margin_change", "eps_growth"} <= set(out)
    assert out["revenue_growth"] == pytest.approx(12.5 / 11.1 - 1)
