import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer


# Download `nltk` dependencies to the `tmp` directory, as there are
# permission issues otherwise:
# http://lab.astamuse.co.jp/entry/2020/04/08/113000
NLTK_DATA_DIR: str = '/tmp/nltk_data'
nltk.data.path.append(NLTK_DATA_DIR)

# Globally define SnowBall stemmers
EN_ST = SnowballStemmer(language='english')
ES_ST = SnowballStemmer(language='spanish')
AR_ST = SnowballStemmer(language='arabic')
FR_ST = SnowballStemmer(language='french')


def penn_to_wn(tag):
    """ Map part-of-speech tag to WordNet tag
    https://stackoverflow.com/a/25544239/10044859
    :param tag: part-of-speech tag to be converted
    :returns: wordnet tag
    """
    if tag in ['JJ', 'JJR', 'JJS']:
        return 'a'
    elif tag in ['NN', 'NNS', 'NNP', 'NNPS']:
        return 'n'
    elif tag in ['RB', 'RBR', 'RBS']:
        return 'r'
    elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
        return 'v'
    return None


def lemmatize(tokens):
    """ Token lemmatization using WordNet leveraging POS tagging
    :param list tokens: tokens to lemmatize
    :returns: List of tokens after applying lemmatization
    :rtype: list
    """

    if tokens is None:
        return None

    # download wordnet lemmatizer if not available on worker node - it should
    # already be downloaded from the initialization action on cluster creation
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger", download_dir=NLTK_DATA_DIR)
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", download_dir=NLTK_DATA_DIR)

    wnl = WordNetLemmatizer()

    tagged_tokens = nltk.pos_tag(tokens)

    # convert `penn` POS tag to WordNet tag
    wn_tagged_tokens = [(t, penn_to_wn(p)) for (t, p) in tagged_tokens]

    # lemmatize tokens that have a part of speech, and are not on the exclusion
    # list
    lemmas = []
    exclusion_list = ["pence", "chose"]
    for (token, pos) in wn_tagged_tokens:
        if token in exclusion_list:
            lemmas.append(token)
        elif pos:
            lemmas.append(wnl.lemmatize(token, pos))
        else:
            lemmas.append(wnl.lemmatize(token))

    return lemmas


def stem(text, lang='en'):
    """ text stemming using SnowballStemmer
    Args:
        text: text to stem
        lang: language to stem in (en, es, fr, or ar)
    Returns:
        text after applying stemmer
    """

    if not text:
        return None

    # if text argument is a string, return a string
    if isinstance(text, str):
        text = text.split()

    if not isinstance(text, list):
        raise Exception("Unsupported text argument; must be string or list")

    if lang == "en":
        stemmed_tokens = [EN_ST.stem(i) for i in text]
    elif lang == "ar":
        stemmed_tokens = [AR_ST.stem(i) for i in text]
    elif lang == "es":
        stemmed_tokens = [ES_ST.stem(i) for i in text]
    elif lang == "fr":
        stemmed_tokens = [FR_ST.stem(i) for i in text]
    # WARNING: unsupported language
    else:
        stemmed_tokens = []

    return ' '.join(stemmed_tokens)
