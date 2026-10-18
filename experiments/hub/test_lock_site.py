import base64
import json
import re

import lock_site


def payload(page: str) -> dict:
    return json.loads(re.search(r'<script type="application/json" id="d">(.*?)</script>', page, re.DOTALL).group(1))


def test_lock_encrypts_every_file(tmp_path):
    (tmp_path / "exp1" / "data").mkdir(parents=True)
    (tmp_path / "index.html").write_text("<html><body><h1>secret hub</h1></body></html>")
    (tmp_path / "exp1" / "index.html").write_text("<html><body>secret pairs <a href='data/s.csv'>csv</a></body></html>")
    (tmp_path / "exp1" / "data" / "s.csv").write_text("a,b\n1,2\n")
    (tmp_path / ".nojekyll").write_text("")
    assert lock_site.lock(tmp_path, "hunter2", iterations=1000) == {"locked_pages": 2, "locked_files": 1}
    hub, sub = (tmp_path / "index.html").read_text(), (tmp_path / "exp1" / "index.html").read_text()
    assert "secret" not in hub + sub and lock_site.LOCKED in hub and "Password" in hub
    d = payload(sub)
    key = lock_site.derive("hunter2", base64.b64decode(d["salt"]), d["iter"])
    page = lock_site.open_sealed(key, base64.b64decode(d["page"])).decode()
    assert "secret pairs" in page and 'new URL("../",' in page  # the download helper knows where the site root is
    assert lock_site.open_sealed(key, (tmp_path / "exp1" / "data" / "s.csv").read_bytes()) == b"a,b\n1,2\n"
    assert payload(hub)["salt"] == d["salt"]  # one key for the whole site, so the password is asked once
    assert (tmp_path / ".nojekyll").read_text() == ""


def test_wrong_password_does_not_open(tmp_path):
    (tmp_path / "index.html").write_text("<html><body>x</body></html>")
    lock_site.lock(tmp_path, "right", iterations=1000)
    d = payload((tmp_path / "index.html").read_text())
    key = lock_site.derive("wrong", base64.b64decode(d["salt"]), d["iter"])
    try:
        lock_site.open_sealed(key, base64.b64decode(d["page"]))
    except Exception as err:  # noqa: BLE001
        assert type(err).__name__ == "InvalidTag"
    else:
        raise AssertionError("decrypted with the wrong password")
