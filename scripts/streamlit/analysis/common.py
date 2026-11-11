"""Compatibility imports for shared dashboard data-cleaning helpers."""

import sys
from pathlib import Path


SHARED_MODULES_DIR = Path(__file__).resolve().parents[2] / "shared"
if str(SHARED_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_MODULES_DIR))

from lumhr_common import (  # noqa: E402,F401
    clean_text,
    find_column,
    find_prefix_column,
    minmax_scale,
    normalize_code,
    normalize_key,
    normalize_postcode,
    optional_numeric,
    parse_numeric,
    resolve_base_dir,
    safe_divide,
)
