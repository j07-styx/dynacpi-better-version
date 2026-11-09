"""Resume the pinned November 2026 BindingDB full archive, preserving partial data.
Verify the publisher MD5 before atomically promoting the completed archive.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import time
import urllib.request
import urllib.error
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "bindingdb_202611"
STEM = "BindingDB_All_202611_tsv"
URL = "https://www.bindingdb.org/rwd/bind/downloads/" + STEM + ".zip"
PART = DEST / (STEM + ".tmp")
FINAL = DEST / (STEM + ".zip")
META = DEST / (STEM + ".json")
DEST.mkdir(parents=True, exist_ok=True)

def obtain(url, headers=None, method="GET"):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}, method=method), timeout=90)

def sha_and_md5(path):
    sh = hashlib.sha256()
    md = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            sh.update(chunk)
            md.update(chunk)
    return sh.hexdigest(), md.hexdigest()

def main():
    with obtain(URL, method="HEAD") as r:
        total = int(r.headers["Content-Length"])
    try:
        with obtain(URL + ".md5") as r:
            expected = r.read(1024).decode().strip().split()[0].lower()
        if not re.fullmatch("[0-9a-f]{32}", expected):
            raise RuntimeError("Invalid upstream checksum")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        expected = None
        print(json.dumps({"warning": "Publisher MD5 unavailable (HTTP 404); require full ZIP CRC validation"}), flush=True)
    if FINAL.is_file():
        sha, md = sha_and_md5(FINAL)
        if expected is None or md == expected:
            with zipfile.ZipFile(FINAL) as z:
                bad = z.testzip()
                if bad:
                    raise RuntimeError("Existing ZIP CRC error: " + bad)
            print(json.dumps({"status": "zip_crc_verified_already", "bytes": total, "sha256": sha}), flush=True)
            return
        raise RuntimeError("Existing complete archive checksum mismatches upstream; preserving file")

    start = PART.stat().st_size if PART.exists() else 0
    if start > total:
        raise RuntimeError("Partial file exceeds upstream length")
    print(json.dumps({"status": "resume", "start": start, "total": total}), flush=True)
    errors = 0
    last_reported = start
    while start < total:
        try:
            req_headers = {"Range": "bytes=" + str(start) + "-"}
            with obtain(URL, headers=req_headers) as r:
                cr = r.headers.get("Content-Range", "")
                if r.status != 206 or not cr.startswith("bytes " + str(start) + "-") or not cr.endswith("/" + str(total)):
                    raise RuntimeError("Server did not honor exact byte range; preserving partial archive: " + str((r.status, cr)))
                with PART.open("ab") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        start += len(chunk)
                        if start - last_reported >= 40 * 1024 * 1024:
                            print(json.dumps({"status": "downloading", "bytes": start, "total": total}), flush=True)
                            last_reported = start
            errors = 0
        except (OSError, TimeoutError) as e:
            errors += 1
            print(json.dumps({"status": "retry", "bytes": start, "attempt": errors, "error": str(e)}), flush=True)
            if errors > 5:
                raise
            time.sleep(min(errors * 2, 10))
    if PART.stat().st_size != total:
        raise RuntimeError("Download size did not match expected size")
    sha, md = sha_and_md5(PART)
    if expected is not None and md != expected:
        raise RuntimeError("Download checksum mismatch; preserving partial archive")
    with zipfile.ZipFile(PART) as z:
        info = [{"name": x.filename, "uncompressed_bytes": x.file_size} for x in z.infolist()]
        if not info:
            raise RuntimeError("Invalid zip archive")
        bad = z.testzip()
        if bad:
            raise RuntimeError("ZIP CRC error in " + bad + "; preserving partial archive")
    os.replace(PART, FINAL)
    out = {"source": URL, "bytes": total, "md5": md, "publisher_md5": expected, "sha256": sha, "members": info, "status": "zip_crc_verified", "note": "Publisher MD5 endpoint returned 404" if expected is None else "Publisher MD5 verified"}
    META.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"status": "zip_crc_verified_complete", "bytes": total, "sha256": sha, "members": info}), flush=True)

if __name__ == "__main__":
    main()
