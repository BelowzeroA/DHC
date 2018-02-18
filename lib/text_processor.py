import re
from yargy.tokenizer import Tokenizer


class TextProcessor:

    def __init__(self, typos = {}):
        self.tokenizer = Tokenizer()
        self.typos = typos

    def extract_words_from_list(self, collection):
        result = []
        for line in collection:
            result.extend(self.extract_words_from_text(line))

        return result

    def extract_words_from_text(self, text):
        words = []
        tokens = self.tokenizer(text)
        for token in tokens:
            if self.is_punct(token):
                continue
            if len(str(token.value)) == 1:
                continue
            if isinstance(token.value, str):
                word = token.forms[0].normalized
                if word in self.typos:
                    words.append(self.typos[word])
                else:
                    words.append(word)
            else:
                words.append(str(token.value))

        return words

    def is_punct(self, token):
        return 'PUNCT' in token.forms[0].grammemes


