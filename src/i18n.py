"""
Internationalization (i18n) Engine for 2D to 3D Studio.
Loads translation files dynamically from locales/<lang>.json with fallback to English (en).
Allows users to contribute new language files seamlessly by dropping them into locales/.
"""
import os
import sys
import json
import locale
from pathlib import Path
from typing import Dict, Any, List, Optional

LOCALES_DIR = Path(__file__).resolve().parent.parent / "locales"

# In-memory translation cache: {"en": {...}, "es": {...}}
_TRANSLATION_CACHE: Dict[str, Dict[str, Any]] = {}
_TRANSLATION_MTIMES: Dict[str, int] = {}
_DEFAULT_LANG = "en"
_ACTIVE_LANG = "en"

def detect_system_language() -> str:
    """
    Detects user's OS language environment. Returns 'es' if Spanish, otherwise defaults to 'en'.
    """
    env_lang = os.environ.get("LANG", "") or os.environ.get("LC_ALL", "")
    if not env_lang:
        try:
            loc = locale.getdefaultlocale()
            if loc and loc[0]:
                env_lang = loc[0]
        except Exception:
            pass

    env_lang = env_lang.lower()
    if env_lang.startswith("es"):
        return "es"
    return "en"

def get_available_locales() -> List[Dict[str, str]]:
    """
    Scans the locales directory and returns metadata for all installed translation files.
    """
    available = []
    if not LOCALES_DIR.exists():
        return [{"code": "en", "name": "English", "native_name": "English"}]

    for p in sorted(LOCALES_DIR.glob("*.json")):
        code = p.stem
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                meta = data.get("__meta__", {})
                name = meta.get("name", code.upper())
                native_name = meta.get("native_name", name)
                available.append({
                    "code": code,
                    "name": name,
                    "native_name": native_name,
                    "file": p.name
                })
        except Exception as e:
            available.append({
                "code": code,
                "name": code.upper(),
                "native_name": code.upper(),
                "file": p.name
            })
    return available

def load_locale(lang_code: str) -> Dict[str, Any]:
    """
    Loads JSON translation strings for a language into cache.
    """
    target_file = LOCALES_DIR / f"{lang_code}.json"
    if not target_file.exists():
        # Fallback to default if requested language file doesn't exist
        target_file = LOCALES_DIR / f"{_DEFAULT_LANG}.json"

    if target_file.exists():
        try:
            mtime = target_file.stat().st_mtime_ns
            if (lang_code in _TRANSLATION_CACHE
                    and _TRANSLATION_MTIMES.get(lang_code) == mtime):
                return _TRANSLATION_CACHE[lang_code]
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                _TRANSLATION_CACHE[lang_code] = data
                _TRANSLATION_MTIMES[lang_code] = mtime
                return data
        except Exception as e:
            print(f"[i18n Warning] Could not parse locale {target_file}: {e}", file=sys.stderr)

    _TRANSLATION_CACHE[lang_code] = {}
    return {}

def set_active_language(lang_code: str) -> None:
    """Sets the global active language for backend / CLI."""
    global _ACTIVE_LANG
    _ACTIVE_LANG = lang_code
    load_locale(lang_code)

def get_active_language() -> str:
    """Returns the current active language code."""
    return _ACTIVE_LANG

def t(key: str, lang: Optional[str] = None, **kwargs) -> str:
    """
    Translates a dotted key (e.g. 'presets.anime_name') into the target language.
    Falls back to English if the key is missing in the target language.
    """
    target_lang = lang or _ACTIVE_LANG
    translations = load_locale(target_lang)

    # 1. Look up in target language
    val = _lookup_key(translations, key)

    # 2. Fallback to default language if missing
    if val is None and target_lang != _DEFAULT_LANG:
        default_translations = load_locale(_DEFAULT_LANG)
        val = _lookup_key(default_translations, key)

    # 3. If still missing, return the key itself
    if val is None:
        val = key

    # 4. Interpolate kwargs if provided
    if kwargs and isinstance(val, str):
        try:
            return val.format(**kwargs)
        except Exception:
            return val

    return str(val)

def _lookup_key(data: Dict[str, Any], key: str) -> Optional[Any]:
    parts = key.split(".")
    curr = data
    for p in parts:
        if isinstance(curr, dict) and p in curr:
            curr = curr[p]
        else:
            return None
    return curr
