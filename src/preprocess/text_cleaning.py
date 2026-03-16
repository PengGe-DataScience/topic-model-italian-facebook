import re
from collections.abc import Iterable

import emoji
import ftfy

# Precompile regex patterns (speed matters at 5M docs)
RE_HTML = re.compile(r"<[^>]+>")
RE_URL = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
RE_MENTION = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#(\w+)")
RE_WHITESPACE = re.compile(r"\s+")
RE_REPEAT_CHARS = re.compile(r"(.)\1{3,}")  # aaaa -> aaa


def clean_text_basic(
    text: str | None,
    *,
    replace_urls: bool = True,
    replace_mentions: bool = True,
    normalize_hashtags: bool = True,
    keep_emojis_as_text: bool = True,
    lowercase: bool = True,
) -> str:
    if text is None:
        return ""

    # Ensure type and fix weird encoding artifacts
    text = ftfy.fix_text(str(text))

    # Strip basic HTML
    text = RE_HTML.sub(" ", text)

    # URLs & mentions
    if replace_urls:
        text = RE_URL.sub(" <URL> ", text)
    if replace_mentions:
        text = RE_MENTION.sub(" <MENTION> ", text)

    # Hashtags: keep the word, drop "#"
    if normalize_hashtags:
        text = RE_HASHTAG.sub(r"\1", text)

    # Emojis -> tokens (":red_heart:" -> "emoji_red_heart")
    if keep_emojis_as_text:
        text = emoji.demojize(text, delimiters=(" ", " "))
        # Convert :face_with_tears_of_joy: into "emoji_face_with_tears_of_joy"
        text = text.replace(":", "")
        text = re.sub(r"\b([a-z0-9_]+)\b", r"emoji_\1", text)

    # Normalize repeated characters and whitespace
    text = RE_REPEAT_CHARS.sub(r"\1\1\1", text)
    text = RE_WHITESPACE.sub(" ", text).strip()

    if lowercase:
        text = text.lower()

    return text


def batch_clean_texts(texts: Iterable[str | None], **kwargs) -> list[str]:
    return [clean_text_basic(t, **kwargs) for t in texts]
