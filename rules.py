"""Cheap deterministic pre-filter run before BitNet. Returns a spam verdict or None.

Catches the two patterns the 2B model misses most on real contact-form leads:
  * random-character bot posts ("c7ty50", "kKEUPBJlj", "qZJexV HucdQz Jec")
  * a bot that posts "I wanted to know your price" in dozens of languages
Real leads are Vietnamese or English. Guards keep accent-less Vietnamese
("Goi lai cho toi sau 6h nhe", which langdetect calls Welsh) away from the language rule.
CJK / Hangul are left to the model: foreign buyers in those languages are plausible.
"""

from __future__ import annotations

import re
import unicodedata

try:
    from langdetect import DetectorFactory, LangDetectException, detect_langs

    DetectorFactory.seed = 0  # deterministic
except ImportError:  # language rule is skipped if the package is missing
    detect_langs = None

ALLOWED_LANGS = {"vi", "en"}
LANG_MIN_PROB = 0.95

# Common Vietnamese syllables, written without accents. Two or more in a message mean
# "probably Vietnamese typed without diacritics", so the language rule must not fire.
VI_SYLLABLES = set(
    """a ah anh ban bao biet can cho chi chua co con cua cuoi da dang dat de den di du duoc em
    gia gio goi gui hang hay ho hoi khong ko lai lam lien luc may minh mot mua muon nao nay
    nha nhe nhieu nhu nua o oi phong qua ra roi sao sau tai thang the thi thue toi tren tu
    tuan va vay ve voi xem xin""".split()
)

# Scripts that real leads on these sites never use (CJK and Hangul deliberately excluded).
FOREIGN_SCRIPTS = ("GREEK", "CYRILLIC", "ARMENIAN", "GEORGIAN", "HEBREW", "ARABIC", "BENGALI",
                   "DEVANAGARI", "GUJARATI", "GURMUKHI", "TAMIL", "TELUGU", "KANNADA",
                   "MALAYALAM", "SINHALA", "MYANMAR", "ETHIOPIC", "THAI", "LAO", "KHMER")

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_RANDOM_TOKEN = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{5,16}$")  # exw7fv, c7ty50


_CAMEL = re.compile(r"^(?:[A-Z][a-z]+)+[A-Z]?[a-z]*$")  # NguyenVanA, MasteriCentrePoint: names
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


def _has_inner_caps(tok: str) -> bool:
    """qZJexV, kKEUPBJlj: two+ uppercase letters after the first, mixed with lowercase."""
    return tok.isalpha() and tok.isascii() and any(c.islower() for c in tok) \
        and sum(c.isupper() for c in tok[1:]) >= 2 and not _CAMEL.match(tok)


# Common English words; a Latin-script message with none of these and nothing Vietnamese
# in it is in some other language (langdetect has no Basque, Galician, Luxembourgish...).
EN_WORDS = set(
    """a about am an and any apartment are at available be buy call can could dear do for from get
    have hello hey hi how i in info information interested is it know like looking me my need of
    on or please price project rent sale send some thank thanks the this to unit visit want we what
    when where would you your""".split()
)
_VI_CHARS = re.compile(r"[ăâđêôơưĂÂĐÊÔƠƯ\u0300-\u0323]|[àáạảãèéẹẻẽìíịỉĩòóọỏõùúụủũỳýỵỷỹ]", re.IGNORECASE)
_EMAIL_URL = re.compile(r"\S+@\S+|https?://\S+|www\.\S+")


def _foreign_script(text: str) -> str | None:
    """Dominant non-Latin script. Needs >= 30% of letters so mojibake ("IвЂ™ll") doesn't count."""
    letters = [ch for ch in text if ch.isalpha()]
    counts: dict[str, int] = {}
    for ch in letters:
        if ord(ch) > 0x2FF:
            name = unicodedata.name(ch, "")
            for script in FOREIGN_SCRIPTS:
                if name.startswith(script):
                    counts[script] = counts.get(script, 0) + 1
                    break
    if not counts:
        return None
    script, n = max(counts.items(), key=lambda kv: kv[1])
    return script.title() if n / len(letters) >= 0.3 else None


def check(message: str) -> dict | None:
    text = message.strip()
    tokens = _WORD.findall(text)
    if not tokens:
        return None

    if len(tokens) == 1 and (_RANDOM_TOKEN.match(tokens[0]) or _has_inner_caps(tokens[0])):
        return {"rule": "random_string", "reason": "Random characters, not a real message."}
    long_tokens = [t for t in tokens if len(t) >= 3]
    if len(long_tokens) >= 2 and sum(map(_has_inner_caps, long_tokens)) / len(long_tokens) >= 0.5:
        return {"rule": "random_string", "reason": "Random characters, not a real message."}

    script = _foreign_script(text)
    if script:
        return {"rule": "foreign_script",
                "reason": f"Written in {script} script; real leads are Vietnamese or English (multilingual bot pattern)."}

    if _CJK.search(text) or _VI_CHARS.search(text):
        return None  # CJK left to the model; Vietnamese diacritics mean Vietnamese
    raw_words = [t for t in _WORD.findall(_EMAIL_URL.sub(" ", text)) if not t.isdigit()]
    words = [t.lower() for t in raw_words]
    if sum(t in VI_SYLLABLES for t in words) >= 2:
        return None  # accent-less Vietnamese
    mostly_lowercase = sum(t[0].islower() for t in raw_words) * 2 >= len(raw_words)  # not a Title Case name
    if len(words) >= 4 and mostly_lowercase and not any(t in EN_WORDS or t in VI_SYLLABLES for t in words):
        return {"rule": "foreign_language",
                "reason": "Not written in Vietnamese or English (multilingual bot pattern)."}
    if detect_langs is not None and len(tokens) >= 3 and mostly_lowercase:
        try:
            best = detect_langs(text)[0]
        except LangDetectException:
            return None
        if best.lang not in ALLOWED_LANGS and best.prob >= LANG_MIN_PROB:
            return {"rule": "foreign_language",
                    "reason": f"Written in another language ({best.lang}); real leads are Vietnamese or English (multilingual bot pattern)."}
    return None
