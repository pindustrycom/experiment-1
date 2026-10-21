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
    assert r["old_addresses_kept"] == 4 and r["exp2"] is False
    hub = (tmp_path / "index.html").read_text()
    assert "first nightly run" in hub and "Not published yet." in hub
    # old addresses redirect into /exp1/, data files are still served at the old path
    assert 'url=exp1/about.html' in (tmp_path / "about.html").read_text()
    assert 'url=../exp1/pairs/ko-pep.html' in (tmp_path / "pairs" / "ko-pep.html").read_text()
    assert (tmp_path / "data" / "scores.csv").read_text() == "a,b\n1,2\n"
    assert "Disallow: /" in (tmp_path / "robots.txt").read_text()


def test_every_page_gets_one_nav_with_the_right_depth(tmp_path):
    make_exp1(tmp_path)
    build_hub.build(tmp_path)
    r = build_hub.build(tmp_path)   # a second build replaces the bar instead of adding another
    deep = (tmp_path / "exp1" / "pairs" / "ko-pep.html").read_text()
    assert deep.count('id="labnav"') == 1
    assert 'href="../../index.html"' in deep and 'href="../../exp4/index.html"' in deep
    assert 'href="../../exp1/index.html" aria-current="page"' in deep
    # experiments without pages are left out of the bar
    assert "exp2/index.html" not in deep and "exp7/index.html" not in deep
    hub = (tmp_path / "index.html").read_text()
    assert hub.count('id="labnav"') == 1 and 'class="ln-home" href="index.html" aria-current="page"' in hub
    six = (tmp_path / "exp6" / "index.html").read_text()
    assert six.index('id="labnav"') < six.index("<h1>")
    for d in ("exp1", "exp4", "exp5", "exp6", "exp6_live"):
        assert f'href="../{d}/index.html"' in six
    # redirect pages stay plain
    assert "labnav" not in (tmp_path / "about.html").read_text()
    assert r["pages_with_nav"] == 3 + 5 + 1


def test_nav_links_exp2_and_exp7_once_published(tmp_path):
    make_exp1(tmp_path)
    for d in ("exp2", "exp7"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "index.html").write_text(HEAD.format(r="", b=d))
    build_hub.build(tmp_path)
    page = (tmp_path / "exp1" / "index.html").read_text()
    assert 'href="../exp2/index.html">Exp2 volatility</a>' in page and 'href="../exp7/index.html">Exp7 VWAP</a>' in page
    assert 'href="../exp2/index.html" aria-current="page"' in (tmp_path / "exp2" / "index.html").read_text()


def test_static_pages_copied_with_shared_script_inlined(tmp_path):
    make_exp1(tmp_path)
    build_hub.build(tmp_path)
    four = (tmp_path / "exp4" / "index.html").read_text()
    assert "<title>Resonance Playground</title>" in four and "Joaquin Philco" in four
    assert '<script src="../hub/gl_runtime.js"></script>' not in four and "window.LabGL = { start, dispose }" in four
    assert 'integrity="sha384-' in four
    five = (tmp_path / "exp5" / "index.html").read_text()
    six = (tmp_path / "exp6" / "index.html").read_text()
    live = (tmp_path / "exp6_live" / "index.html").read_text()
    assert "<title>Exp5 Vision Stack</title>" in five and "Joaquin Philco" in five
    assert "<title>Exp6 Alphabet and Words</title>" in six and 'id="scene3d"' in six and "window.LabGL" in six
    assert "<title>Exp6 Live Model</title>" in live and 'id="engine"' in live and "window.LabGL" in live
    eight = (tmp_path / "exp8" / "index.html").read_text()
    assert "<title>Exp8 Resonant Neuron</title>" in eight and 'id="voldata"' in eight and "window.LabGL" in eight
    hub = (tmp_path / "index.html").read_text()
    for d, title in (("exp4", "Exp4 · resonance playground"), ("exp5", "Exp5 · vision stack"),
                     ("exp6", "Exp6 · alphabet and words"), ("exp6_live", "Exp6 live · the model in your browser")):
        assert f'href="{d}/index.html"' in hub and title in hub


def test_source_pages_load_the_shared_runtime_locally():
    # opened from the repository, ../hub/gl_runtime.js resolves to this folder
    for d in ("exp4_resonance", "exp6_alphabet", "exp6_live", "exp8_neuron"):
        page = (build_hub.EXPERIMENTS / d / "index.html").read_text()
        assert '<script src="../hub/gl_runtime.js"></script>' in page and "LabGL.start(" in page
        assert "requestAnimationFrame(frame)" not in page
    assert (build_hub.HUB / "gl_runtime.js").exists()


def test_hub_without_exp4(tmp_path):
    assert build_hub.add_exp4(tmp_path, tmp_path / "missing") is False
    (tmp_path / "index.html").write_text(build_hub.hub_page(tmp_path))
    assert "Exp4" not in (tmp_path / "index.html").read_text()


def test_overview_is_one_column_per_group(tmp_path):
    make_exp1(tmp_path)
    build_hub.build(tmp_path)
    hub = (tmp_path / "index.html").read_text()
    assert hub.count('class="half"') == len(build_hub.GROUPS)
    for _g, anchor, *_ in build_hub.GROUPS:
        assert f'id="{anchor}"' in hub and f'class="door" href="#{anchor}"' in hub
    assert 'class="ln-g" href="index.html#resonance">Resonance</a>' in hub
    assert 'href="../index.html#markets"' in (tmp_path / "exp1" / "index.html").read_text()
