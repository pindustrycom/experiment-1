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
    assert "first run" in (tmp_path / "index.html").read_text()
    # old addresses redirect into /exp1/, data files are still served at the old path
    assert 'url=exp1/about.html' in (tmp_path / "about.html").read_text()
    assert 'url=../exp1/pairs/ko-pep.html' in (tmp_path / "pairs" / "ko-pep.html").read_text()
    assert (tmp_path / "data" / "scores.csv").read_text() == "a,b\n1,2\n"
    # exp1 pages link back to the hub with the right depth, once
    assert 'href="../../index.html"' in (tmp_path / "exp1" / "pairs" / "ko-pep.html").read_text()
    assert 'href="../index.html" style' in (tmp_path / "exp1" / "index.html").read_text()
    build_hub.build(tmp_path)
    assert (tmp_path / "exp1" / "index.html").read_text().count("Paper portfolio") == 1


def test_exp1_links_to_exp2(tmp_path):
    make_exp1(tmp_path)
    build_hub.build(tmp_path)
    assert 'href="../exp2/index.html"' in (tmp_path / "exp1" / "index.html").read_text()
    assert 'href="../../exp2/index.html"' in (tmp_path / "exp1" / "pairs" / "ko-pep.html").read_text()


def test_exp4_page_copied_and_linked(tmp_path):
    make_exp1(tmp_path)
    build_hub.build(tmp_path)
    page = (tmp_path / "exp4" / "index.html").read_text()
    assert "<title>Resonance Playground</title>" in page and "Joaquin Philco" in page
    hub = (tmp_path / "index.html").read_text()
    assert 'href="exp4/index.html">Experiment 4</a>' in hub and "Exp4 · resonance playground" in hub


def test_exp5_and_exp6_pages_copied_and_linked(tmp_path):
    make_exp1(tmp_path)
    build_hub.build(tmp_path)
    five = (tmp_path / "exp5" / "index.html").read_text()
    six = (tmp_path / "exp6" / "index.html").read_text()
    assert "<title>Exp5 Vision Stack</title>" in five and "Joaquin Philco" in five
    assert "<title>Exp6 Alphabet and Words</title>" in six and "Joaquin Philco" in six and 'id="scene3d"' in six
    assert 'href="../exp6/index.html"' in five and 'href="../exp5/index.html"' in six
    hub = (tmp_path / "index.html").read_text()
    for n in (4, 5, 6):
        assert f'href="exp{n}/index.html">Experiment {n}</a>' in hub
    assert "Exp5 · vision stack" in hub and "Exp6 · alphabet and words" in hub


def test_exp6_live_page_copied_and_linked(tmp_path):
    make_exp1(tmp_path)
    build_hub.build(tmp_path)
    live = (tmp_path / "exp6_live" / "index.html").read_text()
    assert "<title>Exp6 Live Model</title>" in live and 'id="engine"' in live and "Joaquin Philco" in live
    assert 'href="../exp6_live/index.html"' in (tmp_path / "exp6" / "index.html").read_text()
    hub = (tmp_path / "index.html").read_text()
    assert 'href="exp6_live/index.html">Experiment 6 live</a>' in hub and "Exp6 live · the model in your browser" in hub


def test_hub_without_exp4(tmp_path):
    assert build_hub.add_exp4(tmp_path, tmp_path / "missing") is False
    (tmp_path / "index.html").write_text(build_hub.hub_page(tmp_path))
    assert "Experiment 4" not in (tmp_path / "index.html").read_text()
