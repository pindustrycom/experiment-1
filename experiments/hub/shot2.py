import sys
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium")
    for name, w, scheme in (("index.html", 1100, "light"), ("index.html", 1100, "dark"), ("exp1/index.html", 1100, "light"), ("index.html", 390, "light")):
        pg = b.new_page(viewport={"width": w, "height": 900}, color_scheme=scheme)
        pg.goto("file://" + sys.argv[1] + "/" + name)
        pg.screenshot(path=f"{sys.argv[1]}/../{name.replace('/','_')}_{w}_{scheme}.png", full_page=True)
    b.close()
