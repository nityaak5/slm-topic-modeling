"""
Convert html_out JSON files to a stance-detection CSV matching
the format of semeval2016_task6_stance.csv / ezstance_test.csv.

Filtering (applied in order):
  1. Malformed records (pandas Series leaked into fields)
  2. Empty content (<300 chars)
  3. Dead/error pages — checked anywhere in content
  4. Boilerplate-only pages — cookie consent / page-load errors starting
     in first 500 chars (language-aware)
  5. Exact content duplicates — keep first occurrence

Output columns:
  id, doc_id, content, query, stance_label, stance_label_original,
  source_dataset, domain, country
  (stance_label / stance_label_original left empty for LLM assignment)
"""

import json
import os
import pandas as pd

FOLDER = os.path.join(os.path.dirname(__file__), "..", "html_out")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data_in", "html_out_stance.csv")

# ── Filter 1: dead / error pages (checked anywhere in content) ────────────────
DEAD_PAGE_PATTERNS = [
    "puzzle",
    "page has disappeared",
    "no longer exists",
    "temporary error",
    "access denied",
    "page not found",
    "the page you are looking for",
    "vissza a főoldalra facebookx",       # Hungarian nav-only dump
    "keresett kifejezés",                 # Hungarian "search term" nav text
    "leider ist die gesuchte seite nicht verfügbar",  # German 404
    "die seite existiert nicht mehr",     # German "page no longer exists"
    "access to this resource on the server is denied",  # server 403
    "litespeed technologies inc.",        # LiteSpeed server-level block page
]

# ── Filter 2: boilerplate-only — matched against first 800 chars ──────────────
BOILERPLATE_LEAD_PATTERNS = [
    # Cookie consent (EN)
    "these cookies may be set through our site by our advertising",
    "cookies, device or similar online identifiers",
    # Czech Cloudflare challenge
    "pro přístup na tento web dokončete bezpečnostní kontrolu",
    # Czech cookie walls
    "na těchto webových stránkách se používají soubory cookies a další síťové identifikátory",
    # Czech Mafra cookie wall
    "mafra a.s., partneři iab",
    # Czech page-load failure
    "zdá se, že se stránka nepodařila správně načíst",
    # Slovak cookie consent
    "s vašim súhlasom, my a naši",
    # German tracking consent wall
    "einwilligungen in die nutzung von tracking-technologien",
    # German 404
    "die angeforderte seite wurde nicht gefunden",
    # Dutch 404
    "de pagina die je zoekt is verwijderd van de website",
    # French cookie walls
    "avec votre accord, nos partenaires et nous utilisons des cookies",
    "pour france télévisions, le respect de votre vie privée est une priorité",
    "si vous choisissez de bénéficier de l'offre payante, aucun cookie",
    # Dutch cookie walls
    "mediahuis (wij) en onze partners gebruiken technologieën zoals cookies",
    "groene.nl gebruikt cookies",
    # Hungarian cookie / subscription walls
    "előfizetés",
    # Generic subscription walls
    "already a subscriber? log in",
    "to continue, please click the box below to let us know you",
    # Slovak/Czech anti-scraping challenge
    "please disable google translate",
    # Vox subscription banner (not article content)
    "when news breaks, you need to understand what actually matters",
    # Cloudflare bot-check page
    "this website uses a security service to protect against malicious bots",
    # French cookie consent variant with partner count between "nos" and "partenaires"
    "avec votre accord, nos",
]


def is_malformed(d: dict) -> bool:
    qt = d.get("query_type")
    return not isinstance(qt, str) or "\n" in qt or "dtype:" in qt


def is_dead_page(content_lower: str) -> bool:
    return any(p in content_lower for p in DEAD_PAGE_PATTERNS)


def is_boilerplate_lead(content: str) -> bool:
    lead = content[:1000].lower()
    return any(p in lead for p in BOILERPLATE_LEAD_PATTERNS)


def load_and_filter():
    rows = []
    skipped = {"malformed": 0, "empty": 0, "dead": 0, "boilerplate": 0}

    for fname in sorted(os.listdir(FOLDER)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(FOLDER, fname)
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            skipped["malformed"] += 1
            continue

        if is_malformed(d):
            skipped["malformed"] += 1
            continue

        content = str(d.get("content") or "").strip()
        cl = content.lower()

        if len(cl) < 500:
            skipped["empty"] += 1
            continue
        if is_dead_page(cl):
            skipped["dead"] += 1
            continue
        if is_boilerplate_lead(content):
            skipped["boilerplate"] += 1
            continue

        rows.append({
            "doc_id": d["_id"],
            "content": content,
            "query": d.get("query", ""),
            "stance_label": "",
            "stance_label_original": "",
            "source_dataset": "html_out",
            "domain": d.get("query_type", ""),
            "country": d.get("country", ""),
        })

    print(f"Pass 1 — malformed: {skipped['malformed']}, empty: {skipped['empty']}, "
          f"dead: {skipped['dead']}, boilerplate: {skipped['boilerplate']}")
    print(f"Before dedup: {len(rows)}")
    return rows


def deduplicate(rows: list[dict]) -> tuple[list[dict], int]:
    seen_content: set[str] = set()
    unique, n_dropped = [], 0
    for row in rows:
        if row["content"] in seen_content:
            n_dropped += 1
        else:
            seen_content.add(row["content"])
            unique.append(row)
    return unique, n_dropped


def main():
    rows = load_and_filter()
    rows, n_dedup = deduplicate(rows)
    print(f"Pass 2 — exact content duplicates dropped: {n_dedup}")
    print(f"Final kept: {len(rows)}")

    df = pd.DataFrame(rows)
    df.insert(0, "id", range(1, len(df) + 1))
    df = df[["id", "doc_id", "content", "query", "stance_label",
             "stance_label_original", "source_dataset", "domain", "country"]]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nSaved → {OUT_PATH}")
    print()
    print("domain distribution:")
    print(df["domain"].value_counts().to_string())
    print()
    print("country distribution:")
    print(df["country"].value_counts().to_string())
    print()
    print("content length (chars) — min/median/max:")
    lens = df["content"].str.len()
    print(f"  min={lens.min()}  median={int(lens.median())}  max={lens.max()}")
    print()
    print("unique queries:", df["query"].nunique())
    print("queries appearing >1 time:", (df["query"].value_counts() > 1).sum())


if __name__ == "__main__":
    main()
