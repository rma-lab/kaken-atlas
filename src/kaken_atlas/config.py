"""プロジェクト共通の設定（パス・秘密情報の読み込み）。"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def load_appid() -> str:
    """KAKEN の appid を環境変数 KAKEN_APP_ID または .env から読む。"""
    value = os.environ.get("KAKEN_APP_ID")
    if value:
        return value
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            if key.strip() == "KAKEN_APP_ID" and val.strip():
                return val.strip()
    raise RuntimeError(
        "KAKEN_APP_ID が見つかりません。プロジェクト直下の .env に "
        "KAKEN_APP_ID=xxxx を設定してください。"
    )
