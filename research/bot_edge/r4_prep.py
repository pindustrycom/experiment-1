"""Round 4: pick the freqtrade strategies to run from the round-4 repos, one file per new class.

  python r4_prep.py OUT results/results.csv REPO_DIR [REPO_DIR ...]

Skips dry-run copies, tests, templates, FreqAI strategies (they need a model-training setup) and every class name
already tested in rounds 1-3 (results.csv) or seen earlier in this round. Writes OUT/<group>/<Class>.py with groups
s1..s7 (long-only, spread over seven CI parts) and f (strategies that short, run as futures), plus OUT/list.csv.
"""
import glob
import os
import re
import shutil
import sys

import pandas as pd

out, results, repos = sys.argv[1], sys.argv[2], sys.argv[3:]
done = set(pd.read_csv(results).strategy.astype(str))
seen, rows = set(), []
for repo in repos:
    for f in sorted(glob.glob(f"{repo}/**/*.py", recursive=True)):
        rel = os.path.relpath(f, repo)
        if any(x in rel for x in ("dry-run/", "tests/", "scripts/", "template", "sample_strategy", "/.git/")):
            continue
        src = open(f, errors="ignore").read()
        m = re.search(r"^class (\w+)\(\s*IStrategy", src, re.M)
        if not m or "freqai" in src.lower():
            continue
        cls = m.group(1)
        if cls in done or cls in seen:
            continue
        seen.add(cls)
        tf = re.search(r"^\s+timeframe\s*=\s*['\"](\w+)['\"]", src, re.M)
        short = bool(re.search(r"^\s+can_short\s*(:\s*bool)?\s*=\s*True", src, re.M))
        rows.append({"repo": os.path.basename(repo.rstrip("/")), "class": cls, "tf": tf.group(1) if tf else "5m",
                     "short": short, "file": rel})
d = pd.DataFrame(rows)
spot = d[~d.short].index
d["group"] = "f"
d.loc[spot, "group"] = [f"s{1 + i % 7}" for i in range(len(spot))]
for r in d.itertuples():
    os.makedirs(f"{out}/{r.group}", exist_ok=True)
    shutil.copy(f"{[x for x in repos if os.path.basename(x.rstrip('/')) == r.repo][0]}/{r.file}", f"{out}/{r.group}/{r._2}.py")
d.to_csv(f"{out}/list.csv", index=False)
print(d.groupby(["repo", "group"]).size().to_string())
