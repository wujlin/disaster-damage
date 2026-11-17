"""Read and write id lists produced by the overlap audit."""

from __future__ import annotations

import json
from pathlib import Path


def read_id_list(path: str | Path | None) -> set[str]:
    if path is None or str(path).strip() in {"", "null", "None"}:
        return set()
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Exclusion file does not exist: {file_path}")
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    if text[0] in "[{":
        payload = json.loads(text)
        if isinstance(payload, list):
            return {str(item) for item in payload}
        if isinstance(payload, dict):
            for key in ("ids", "group_ids", "image_ids", "sagin_group_ids", "xbd_image_ids"):
                if key in payload:
                    return {str(item) for item in payload[key]}
        raise ValueError(f"Exclusion JSON must be a list or an ids object: {file_path}")
    ids = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        ids.add(stripped.split()[0])
    return ids


def write_id_list(path: Path, ids: set[str] | list[str], header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [f"# {header}"]
    body.extend(sorted(set(ids)))
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
