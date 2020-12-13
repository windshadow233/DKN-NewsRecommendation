import unicodedata
import nltk


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    return s


def split_words(words):
    return [normalize_string(s) for s in nltk.word_tokenize(words)]


def get_entities_from_title(title, entities):
    """
    此函数将title每个单词对应到一个entity(若不存在对应则对应到<pad>)
    :param title: 经分词后的title
    :param entities: 对应的entities
    """
    result = ['<pad>' for _ in range(len(title))]
    for entity in entities:
        wikiId = entity.get('WikidataId')
        surface_forms = entity.get('SurfaceForms')
        for form in surface_forms:
            form = split_words(form)
            length = len(form)
            for i in range(len(title) - length + 1):
                if title[i: i + length] == form:
                    result[i: i + length] = [wikiId] * length
    return result

