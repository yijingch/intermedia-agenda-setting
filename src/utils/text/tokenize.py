import pyspark
import emoji
import regex as re  # Regex module with Unicode support

from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize import TweetTokenizer
from pyspark.sql import functions as f
from pyspark.sql.types import ArrayType, StringType
from shared.text import contractions, text_helpers

tweet_tokenizer = TweetTokenizer(preserve_case=False)
toktok = ToktokTokenizer()


def multi_language_tokenizer(text, lang_attr=None):
    """Tokenization for Arabic/Spanish using stanza.

    Note: Tokenizing using stanze slows down the processing time
    by x3. Instead the workaround would be to use nltk.
    On testing it was noted that tweet_tokenizer works appropriately
    for arabic as well.
    For Spanish source: https://github.com/nltk/nltk/issues/1558

    :param text: text to process
    :param lang_attr: lang to process tokennization in
    :return: list of tokens extracted from text
    """
    # spanish tokenization
    if lang_attr == "es":
        return toktok.tokenize(text)

    # default to tweet tokenizer
    else:
        return tweet_tokenizer.tokenize(text)

    # Leaving this here for now
    # nlp = stanza.Pipeline(lang=lang_attr, processors='tokenize', verbose=False)
    # doc = nlp(text)
    # tokens = []

    # for sentence in (doc.sentences):
        #     for token in sentence.tokens:
        # 	    tokens.append(token.text)

    # # print("Tokens -> ", tokens)

    # return tokens


def tokenize(text):
    """Use globally declared tokenizer on text

    >>> columns = ['id', 'text']
    >>> vals = [(1, "Example tokenization"),]
    >>> df = sqlc.createDataFrame(vals, columns)
    >>> df.withColumn("tokens", udfs.tokenize("text")).show()

    :param text: string to tokenize
    :return: list of tokens extracted from text
    """
    return tweet_tokenizer.tokenize(text)


def filter_tokens(tokens, stopwords=[], urls=False, remove_emojis=False, filter_hashtags=False):
    """
    Removed unwanted characters from tokens, by default removing empty
    strings

    NOTE: it would likely be more efficient to pass the stopwords list as an
    argument, or use a broadcast variable, instead of loading it each function
    call...

    :param tokens: list of elements to filter
    :param stopwords: list or broadcasted list of stopwords
    :param urls: filter English stopwords from tokens
    """

    # Empty tokens, and stray apostrophes and colons
    filtered = filter(lambda x: x and x != "'" and x != ":", tokens)

    if isinstance(stopwords, pyspark.broadcast.Broadcast):
        stopwords = stopwords.value

    if stopwords:
        filtered = filter(lambda x: x not in stopwords, filtered)

    # Remove URLs prefixed with `http` and `https`
    if urls:
        # Removing punctuation first breaks this regex, research approaches
        # filtered = filter(lambda t: not re.match(r"(http|https)://.*", t), filtered)
        filtered = filter(lambda t: not re.match(r"^(http|https).*", t), filtered)

    # Remove emojis
    if remove_emojis:
        filtered = filter(lambda t: emoji.get_emoji_regexp().sub(u'', t), filtered)

    # Remove hashtags
    if filter_hashtags:
        filtered = filter(lambda t: not t.startswith("#"), filtered)

    return list(filtered)


@f.udf(returnType=ArrayType(StringType()))
def udf_tokenize(text):
    """Composition of text processing functions: cleanup & tokenization.

    :param text: text to process
    """
    if text:
        text = text.lower()
        text = text_helpers.tweet_text_cleanup(text)
        text = contractions.expand(text)
        text = text_helpers.remove_punctuation(text)
        tokens = tokenize(text)
        tokens = filter_tokens(tokens, urls=False)
        return tokens
    else:
        return []
