"""Look up bibliographic metadata on Crossref (GET only).

Usage: python crossref.py doi [doi ...]
       python crossref.py --q "free-text query" [...]
"""

import json
import sys
import urllib.parse
import urllib.request


def show(m):
    au = ", ".join(
        (a.get("given", "") + " " + a.get("family", "")).strip()
        for a in m.get("author", [])
    )
    print(
        f"{au} | {(m.get('title') or [''])[0]} | "
        f"{(m.get('container-title') or [''])[0]} "
        f"vol={m.get('volume')} iss={m.get('issue')} "
        f"pp={m.get('page') or m.get('article-number')} | "
        f"{m.get('issued', {}).get('date-parts')} | doi={m.get('DOI')}"
    )


args = sys.argv[1:]
i = 0
while i < len(args):
    a = args[i]
    if a == "--q":
        q = args[i + 1]
        i += 2
        url = (
            "https://api.crossref.org/works?rows=3&query.bibliographic="
            + urllib.parse.quote(q)
        )
        with urllib.request.urlopen(url) as r:
            items = json.load(r)["message"]["items"]
        print(f"## {q}")
        for m in items:
            show(m)
    else:
        i += 1
        url = "https://api.crossref.org/works/" + urllib.parse.quote(
            a, safe=""
        )
        try:
            with urllib.request.urlopen(url) as r:
                show(json.load(r)["message"])
        except Exception as e:  # noqa: BLE001
            print("ERR", a, e)
