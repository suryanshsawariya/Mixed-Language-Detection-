from pathlib import Path
from urllib.request import urlopen, Request

BASE = "https://raw.githubusercontent.com/lingo-iitgn/COMI-LINGUA/main/"
OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

for name in ["LID_train.csv", "LID_test.csv"]:
    url = BASE + name
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    print(f"Downloading {name}...")
    with urlopen(req) as r:
        data = r.read()
    (OUT / name).write_bytes(data)
    print(f"Saved: {OUT/name} ({len(data)/1024/1024:.1f} MB)")
print("Dataset download complete.")
