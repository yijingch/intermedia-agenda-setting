from langdetect import detect
from langdetect import DetectorFactory


def detect_language(tokens):
    """Detect language of tokenized text

    :param tokens: tokenized text used in language detection
    :return: detected language of tokens
    """

    DetectorFactory.seed = 0  # enforce consistent language detection results

    if all(token.startswith("http") for token in tokens):
        return "links"

    alpha_tokens = [t for t in tokens if t.isalpha()]
    if not alpha_tokens:
        return "unknown"
    else:
        try:
            return detect(" ".join(alpha_tokens))
        except Exception:
            return "unknown"
