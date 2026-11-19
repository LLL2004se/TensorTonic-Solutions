def remove_stopwords(tokens: list, stopwords: list) -> list:
    stopwords_set = set(stopwords)
    return [token for token in tokens if token not in stopwords_set]