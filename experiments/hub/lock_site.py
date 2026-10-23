"""Put the whole GitHub Pages site behind one password, by encrypting every file in it.

    PASSWORD_SITE=... python experiments/hub/lock_site.py --site _site   (pages.yml passes the PASSWORD secret)

GitHub Pages only serves static files, so there is no server that could check a password. Instead every
published file is encrypted with AES-256-GCM under a key derived from the password (PBKDF2-SHA256, a
random salt per build). Each HTML page is replaced by a small login page that carries its encrypted
original; the browser derives the key from the password the visitor types, decrypts the page and shows
it. A wrong password simply fails to decrypt. The key is kept in sessionStorage, so the password is asked
once per browser tab session. Data files (CSV, JSON, ...) are stored encrypted at the same address and are
decrypted on download by a small script added to each unlocked page.

This is client-side encryption, not a server login: anyone can download the encrypted files, but they
cannot read them without the password. It does not hide anything published elsewhere (the repository,
its releases). Needs the `cryptography` package.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

ITERATIONS = 600_000
LOCKED = "<!--locked-->"
SKIP = {".nojekyll", "robots.txt"}  # served as they are: neither says anything about the site

# Added to each page before it is encrypted: once the page is unlocked, links to the site's own data files
# (which are stored encrypted) are fetched, decrypted with the session key and saved.
HELPER = """<script>(()=>{const S="exp1-site-key",root=new URL({{ROOT}},location.href).href;
document.addEventListener("click",async ev=>{const a=ev.target.closest&&ev.target.closest("a[href]");if(!a||ev.defaultPrevented||ev.button)return;
const u=new URL(a.href,location.href);if(!u.href.startsWith(root)||u.hash&&u.pathname===location.pathname)return;
if(/(\\.html?|\\/)$/i.test(u.pathname))return;ev.preventDefault();
try{const k=JSON.parse(sessionStorage.getItem(S)),key=await crypto.subtle.importKey("raw",Uint8Array.from(atob(k.key),c=>c.charCodeAt(0)),"AES-GCM",false,["decrypt"]);
const b=new Uint8Array(await (await fetch(u.href,{cache:"no-store"})).arrayBuffer());
const p=await crypto.subtle.decrypt({name:"AES-GCM",iv:b.slice(0,12)},key,b.slice(12));
const t={csv:"text/csv",json:"application/json",txt:"text/plain",png:"image/png",svg:"image/svg+xml",pdf:"application/pdf"}[u.pathname.split(".").pop().toLowerCase()]||"application/octet-stream";
const o=URL.createObjectURL(new Blob([p],{type:t})),d=document.createElement("a");d.href=o;d.download=u.pathname.split("/").pop();
document.body.appendChild(d);d.click();d.remove();setTimeout(()=>URL.revokeObjectURL(o),60000)}catch(e){alert("Could not open this file.")}})})();</script>"""

# The login page: pure black, a spinning point-cloud torus behind one password field. A wrong password turns
# the field and the torus red until the visitor types again.
LOGIN = """<!doctype html><html lang="en"><head><meta charset="utf-8">""" + LOCKED + """
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow"><meta name="referrer" content="no-referrer">
<title>Private dashboard</title><style>
:root{color-scheme:dark;--text:#f5f5f7;--line:rgba(255,255,255,.14);--hi:#ff453a}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:#000;overflow:hidden}
body{display:flex;align-items:center;justify-content:center;padding:16px;color:var(--text);
font:16px/1.5 -apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",system-ui,sans-serif;-webkit-font-smoothing:antialiased}
canvas{position:fixed;inset:0;width:100%;height:100%;display:block}
form{position:relative;width:100%;max-width:320px;transition:opacity .3s}body.wait form{opacity:0;pointer-events:none}
input{width:100%;font:inherit;color:var(--text);text-align:center;letter-spacing:.12em;padding:14px 18px;border-radius:999px;
border:1px solid var(--line);outline:none;background:linear-gradient(150deg,rgba(52,26,92,.38),rgba(28,14,52,.22) 45%,rgba(255,255,255,.03));
box-shadow:0 20px 50px rgba(0,0,0,.55),inset 0 1px 0 rgba(255,255,255,.10);-webkit-backdrop-filter:blur(30px) saturate(180%);backdrop-filter:blur(30px) saturate(180%);
transition:border-color .2s,box-shadow .2s,color .2s}
input::placeholder{color:rgba(245,245,247,.38);letter-spacing:.02em}
input:focus{border-color:rgba(165,147,212,.55);box-shadow:0 20px 50px rgba(0,0,0,.55),0 0 0 3px rgba(122,98,179,.22)}
body.bad input{color:var(--hi);border-color:var(--hi);box-shadow:0 0 0 3px rgba(255,69,58,.25),0 0 40px rgba(255,69,58,.25);animation:shake .4s}
body.busy input{opacity:.6}
@keyframes shake{20%,60%{transform:translateX(-8px)}40%,80%{transform:translateX(8px)}}
body.bad input::selection{background:rgba(255,69,58,.35)}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
</style></head><body class="wait"><canvas id="c" aria-hidden="true"></canvas>
<form id="f"><label class="sr" for="p">Password</label>
<input id="p" type="password" name="password" autocomplete="current-password" placeholder="Password" required autofocus></form>
<script type="application/json" id="d">{{DATA}}</script>
<script>(()=>{const S="exp1-site-key",d=JSON.parse(document.getElementById("d").textContent),B=document.body,
u=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0)),f=document.getElementById("f"),p=document.getElementById("p");
// torus: points on the surface, rotated about two axes, perspective-projected, lit by their normal
const cv=document.getElementById("c"),g=cv.getContext("2d"),still=matchMedia("(prefers-reduced-motion: reduce)").matches;let A=0.5,Bt=0,red=0,run=true;
const size=()=>{const r=devicePixelRatio||1;cv.width=innerWidth*r;cv.height=innerHeight*r};size();addEventListener("resize",size);
const draw=()=>{if(!run)return;const W=cv.width,H=cv.height,s=Math.min(W,H)*0.23,ca=Math.cos(A),sa=Math.sin(A),cb=Math.cos(Bt),sb=Math.sin(Bt);
g.fillStyle="#000";g.fillRect(0,0,W,H);red=Math.min(1,Math.max(0,red+(B.classList.contains("bad")?0.08:-0.03)));const m=red;
const R0=Math.round(150+105*m),G0=Math.round(120-51*m),B0=Math.round(220-162*m),dot=Math.max(1,s/110);
for(let i=0;i<6.283;i+=0.075){const ci=Math.cos(i),si=Math.sin(i);for(let j=0;j<6.283;j+=0.03){const cj=Math.cos(j),sj=Math.sin(j),h=2+ci;
const x=h*(cb*cj+sa*sb*sj)-si*ca*sb,y=h*(sb*cj-sa*cb*sj)+si*ca*cb,z=5+ca*h*sj+si*sa,o=1/z;
const L=cj*ci*sb-ca*ci*sj-sa*si+cb*(ca*si-ci*sa*sj);if(L<=0)continue;g.fillStyle=`rgba(${R0},${G0},${B0},${Math.min(0.9,0.06+L*0.5)})`;
g.fillRect(W/2+s*2.6*x*o,H/2-s*2.6*y*o,dot,dot)}}if(!still){A+=0.004;Bt+=0.002}requestAnimationFrame(draw)};draw();
const show=async raw=>{const key=await crypto.subtle.importKey("raw",raw,"AES-GCM",false,["decrypt"]),c=u(d.page);
const html=new TextDecoder().decode(await crypto.subtle.decrypt({name:"AES-GCM",iv:c.slice(0,12)},key,c.slice(12)));
try{sessionStorage.setItem(S,JSON.stringify({salt:d.salt,key:btoa(String.fromCharCode(...raw))}))}catch(_){}
await new Promise(r=>document.readyState==="complete"?r():addEventListener("load",r));
run=false;document.open();document.write(html);document.close()};
const saved=(()=>{try{const k=JSON.parse(sessionStorage.getItem(S));return k&&k.salt===d.salt?u(k.key):null}catch(_){return null}})();
(saved?show(saved):Promise.reject()).catch(()=>{B.classList.remove("wait");p.focus()});
p.addEventListener("input",()=>B.classList.remove("bad"));
f.addEventListener("submit",async ev=>{ev.preventDefault();if(B.classList.contains("busy"))return;B.classList.remove("bad");B.classList.add("busy");
try{const pw=await crypto.subtle.importKey("raw",new TextEncoder().encode(p.value),"PBKDF2",false,["deriveBits"]);
const raw=new Uint8Array(await crypto.subtle.deriveBits({name:"PBKDF2",hash:"SHA-256",salt:u(d.salt),iterations:d.iter},pw,256));
await show(raw)}
catch(_){B.classList.remove("busy");void B.offsetWidth;B.classList.add("bad");p.select()}})})();</script>
<noscript><p style="position:relative;text-align:center;color:#98989f">This page needs JavaScript to unlock.</p></noscript></body></html>"""


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def derive(password: str, salt: bytes, iterations: int = ITERATIONS) -> bytes:
    return PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=iterations).derive(password.encode())


def seal(key: bytes, data: bytes) -> bytes:
    iv = os.urandom(12)
    return iv + AESGCM(key).encrypt(iv, data, None)


def open_sealed(key: bytes, blob: bytes) -> bytes:
    return AESGCM(key).decrypt(blob[:12], blob[12:], None)


def with_helper(page: str, root: str) -> str:
    helper = HELPER.replace("{{ROOT}}", json.dumps(root))
    i = page.lower().rfind("</body>")
    return page[:i] + helper + page[i:] if i >= 0 else page + helper


def login_page(page_blob: bytes, salt: bytes, iterations: int) -> str:
    data = json.dumps({"salt": b64(salt), "iter": iterations, "page": b64(page_blob)})
    return LOGIN.replace("{{DATA}}", data.replace("<", "\\u003c"))


def lock(site: Path, password: str, iterations: int = ITERATIONS) -> dict:
    if not password:
        raise ValueError("empty password")
    salt = os.urandom(16)
    key = derive(password, salt, iterations)
    pages = files = 0
    for f in sorted(p for p in site.rglob("*") if p.is_file()):
        rel = f.relative_to(site)
        if rel.name in SKIP:
            continue
        if f.suffix.lower() in (".html", ".htm"):
            page = f.read_text(encoding="utf-8")
            if LOCKED in page:
                continue
            root = "../" * (len(rel.parts) - 1) or "./"
            f.write_text(login_page(seal(key, with_helper(page, root).encode()), salt, iterations), encoding="utf-8")
            pages += 1
        else:
            f.write_bytes(seal(key, f.read_bytes()))
            files += 1
    return {"locked_pages": pages, "locked_files": files}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", type=Path, default=Path("_site"))
    a = ap.parse_args()
    pw = os.environ.get("PASSWORD_SITE", "")
    if not pw:
        sys.exit("PASSWORD_SITE is not set (pages.yml fills it from the PASSWORD secret): refusing to publish the site without its password.")
    print(json.dumps(lock(a.site, pw)))
