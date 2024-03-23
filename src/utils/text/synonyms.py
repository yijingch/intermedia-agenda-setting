
import re


def replace(text, synonym_dict):
    """ Replace synonyms with their counterparts
    :param text: input text
    :param synonym_dict: key of word, value of synonym
    :return: text with synonyms replaced
    """

    # CRITICAL: it's important to sort the dictionary by key length before
    # forming the regular expression. This is to accomodate for scenarios where
    # a key may be contained within another key.
    #
    # For example, "barack" is contained within "barack obama". Therefore, if a
    # replacement ocurred in the wrong order for the sentence, "the president,
    # barack obama", the result would be "the president, obama obama".
    #
    # TODO: it is very inefficient to be sorting this within ever function
    # call, it should be moved elsewhere...
    synonym_dict = {
        key: value
        for key, value
        in sorted(synonym_dict.items(), key=lambda item: len(item[0]), reverse=True)
    }

    def replace(match):
        return synonym_dict[match.group(0)]

    c_re = re.compile(r"\b(" + "|".join(synonym_dict.keys()) + r")\b")
    return c_re.sub(replace, text)
