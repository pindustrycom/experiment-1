import json

import build_hub

HEAD = '<html><body><header><div class="wrap"><a class="brand" href="{r}index.html">Splittability Monitor</a><nav><a href="{r}index.html">Pairs</a></nav></div></header><main>{b}</main></body></html>'


def make_exp1(site):
    d = site / "exp1"
    (d / "pairs").mkdir(parents=True)
    (d / "data").mkdir()
    (d / "index.html").write_text(HEAD.format(r="", b="pairs"))
    (d / "about.html").write_text(HEAD.format(r="", b="about"))
    (d / "pairs" / "ko-pep.html").write_text(HEAD.format(r="../", b="KO PEP"))
    (d / "data" / "scores.csv").write_text("a,b\n1,2\n")
    (d / "data" / "pairs.json").write_text(json.dumps([
        {"slug": "ko-pep", "company_a": {"ticker": "KO"}, "company_b": {"ticker": "PEP"}, "si_v2": {"value": 61.2}},
        {"slug": "v-ma", "company_a": {"ticker": "V"}, "company_b": {"ticker": "MA"}, "si_v2": None}]))


def test_hub_with_exp1_only(tmp_path):
    make_exp1(tmp_path)
    r = build_hub.build(tmp_path)
    assert r == {"exp1_pages_linked": 3, "old_addresses_kept": 4, "exp2": False}
    hub = (tmp_path / "index.html").read_text()
    assert "KO/PEP 61" in hub and "Not published yet" in hub and 'href="exp1/index.html"' in hub
    # old addresses redirect into /exp1/, data files are still served at the old path
    assert 'url=exp1/about.html' in (tmp_path / "about.html").read_text()
    assert 'url=../exp1/pairs/ko-pep.html' in (tmp_path / "pairs" / "ko-pep.html").read_text()
    assert (tmp_path / "data" / "scores.csv").read_text() == "a,b\n1,2\n"
    # exp1 pages link back to the hub with the right depth, once
    assert 'href="../../index.html"' in (tmp_path / "exp1" / "pairs" / "ko-pep.html").read_text()
    assert 'href="../index.html" style' in (tmp_path / "exp1" / "index.html").read_text()
    build_hub.build(tmp_path)
    assert (tmp_path / "exp1" / "index.html").read_text().count("All experiments") == 1


def test_hub_shows_exp2_summary(tmp_path):
    d = tmp_path / "exp2" / "data"
    d.mkdir(parents=True)
    (tmp_path / "exp2" / "index.html").write_text("x")
    (d / "summary.json").write_text(json.dumps({
        "latest": {"date": "2026-10-16", "exp_ret": 0.081},
        "performance_since_2019": {"portfolio": {"annual_pct": 17.8, "max_dd_pct": 12.3}, "spy": {"annual_pct": 17.3}},
        "calibration_live": {"n": 0}, "calibration_last_120": {"n": 120, "in68": 0.74}}))
    build_hub.build(tmp_path)
    hub = (tmp_path / "index.html").read_text()
    assert "+0.08%" in hub and "74% of 120" in hub and "+17.8% a year" in hub and "12%" in hub and "Not published yet" in hub


def test_exp1_links_to_exp2(tmp_path):
    make_exp1(tmp_path)
    build_hub.build(tmp_path)
    assert 'href="../exp2/index.html"' in (tmp_path / "exp1" / "index.html").read_text()
    assert 'href="../../exp2/index.html"' in (tmp_path / "exp1" / "pairs" / "ko-pep.html").read_text()
