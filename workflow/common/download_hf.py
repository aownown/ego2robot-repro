"""Download only named model files at a fixed revision, using the user's HF login."""
import argparse
import hashlib
import json
import os
from pathlib import Path

# This workstation also sets ALL_PROXY=socks://..., which httpx rejects even
# when HTTP(S)_PROXY is usable. Preserve the explicit HTTP/HTTPS proxy routes.
for key in ("ALL_PROXY", "all_proxy"):
    if os.environ.get(key, "").startswith("socks://") and (
        os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    ):
        os.environ.pop(key)
from huggingface_hub import hf_hub_download

p = argparse.ArgumentParser()
p.add_argument("repo")
p.add_argument("revision")
p.add_argument("destination", type=Path)
p.add_argument("files", nargs="+")
a = p.parse_args()
records = []
for name in a.files:
    path = Path(hf_hub_download(repo_id=a.repo, revision=a.revision,
                              filename=name, local_dir=a.destination))
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
    records.append(dict(file=name, bytes=path.stat().st_size, sha256=digest.hexdigest()))
    print(name, path.stat().st_size, "bytes; SHA256", digest.hexdigest(), flush=True)
(a.destination / "download_manifest.json").write_text(json.dumps(
    dict(repo=a.repo, revision=a.revision, files=records), indent=2) + "\n")
