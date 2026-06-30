#!/usr/bin/env python3
"""
Download real labeled waterfalls from the SatNOGS Network open API.

Every SatNOGS observation has a waterfall image and a human vetting label. We use
`waterfall_status`:  with-signal -> good,  without-signal -> bad. Read access is
public (no API key needed). Data is CC-BY-SA (credit SatNOGS / Libre Space Foundation).

The result is an ImageFolder layout the rest of the pipeline expects:
    data/good/<obs_id>.png
    data/bad/<obs_id>.png
plus manifest.csv.

Usage:
    pip install requests
    python fetch.py --per-class 800           # ~800 of each, good for a first model
    python fetch.py --per-class 2000 --out data
"""
import argparse, csv, os, time
import requests

API = "https://network.satnogs.org/api/observations/"
UA = {"User-Agent": "satnogs-waterfall-classifier/1.0 (portfolio project)",
      "Accept": "application/json"}
# The API's `status` filter takes good/bad/failed/unknown. We map the two we want.
# (good = a signal was vetted present, bad = vetted empty.) `waterfall_status` also
# exists but is a boolean; `status` is the clearest, unambiguous filter.
LABEL_MAP = {"good": "good", "bad": "bad"}


def get_json(url, params, max_retries=6):
    """GET that waits out 429 rate limits instead of crashing."""
    delay = 5
    for _ in range(max_retries):
        r = requests.get(url, headers=UA, timeout=40, params=params)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", delay))
            print(f"  rate-limited (429); waiting {wait}s and retrying...")
            time.sleep(wait)
            delay = min(delay * 2, 60)
            continue
        r.raise_for_status()
        return r
    raise SystemExit("Still rate-limited after several retries. "
                     "Wait a few minutes and re-run, or raise --sleep.")


def fetch_class(status_value, label, out_dir, target, sleep):
    """Page through observations of one status using the API's Link-header
    pagination, downloading each waterfall image (skipping ones with none)."""
    saved = 0
    d = os.path.join(out_dir, label)
    os.makedirs(d, exist_ok=True)
    rows = []
    next_url = API
    params = {"status": status_value}        # sent only on the first request
    while next_url and saved < target:
        r = get_json(next_url, params)
        data = r.json()
        batch = data["results"] if isinstance(data, dict) and "results" in data else data
        if not batch:
            break
        for obs in batch:
            wf = obs.get("waterfall")
            if not wf:
                continue                     # observation has no waterfall image
            path = os.path.join(d, f"{obs['id']}.png")
            if not os.path.exists(path):
                try:
                    img = requests.get(wf, headers=UA, timeout=40)
                    img.raise_for_status()
                    with open(path, "wb") as fh:
                        fh.write(img.content)
                except requests.RequestException:
                    continue
            rows.append((obs["id"], label, obs.get("norad_cat_id"),
                         obs.get("ground_station"), wf))
            saved += 1
            if saved % 25 == 0:
                print(f"  {label}: {saved}/{target}")
            if saved >= target:
                break
        params = None                        # the `next` URL already carries the query
        next_url = r.links.get("next", {}).get("url")
        time.sleep(sleep)                    # be polite to a volunteer-run service
    print(f"{label}: saved {saved}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=800)
    ap.add_argument("--out", default="data")
    ap.add_argument("--sleep", type=float, default=1.0)
    a = ap.parse_args()

    print("Fetching real SatNOGS waterfalls (public API, no key needed)...")
    rows = []
    for status_value, label in LABEL_MAP.items():
        rows += fetch_class(status_value, label, a.out, a.per_class, a.sleep)

    with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["obs_id", "label", "norad_cat_id", "ground_station", "waterfall_url"])
        w.writerows(rows)
    print(f"\nWrote {len(rows)} images + manifest.csv to {a.out}/")
    print("Next:  python train.py --arch resnet18 --epochs 8")
    print("Data: SatNOGS / Libre Space Foundation, CC-BY-SA — credit them if you publish.")


if __name__ == "__main__":
    main()