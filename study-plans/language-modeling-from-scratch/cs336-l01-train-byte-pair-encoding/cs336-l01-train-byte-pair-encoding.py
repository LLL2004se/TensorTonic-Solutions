
def train_bpe(corpus: list[str], vocab_size: int) -> dict:
    # Convert each string into UTF-8 byte token IDs
    words = [list(s.encode("utf-8")) for s in corpus]

    vocab = []
    merges = []

    # Store the byte sequence represented by every token
    token_bytes = {i: bytes([i]) for i in range(256)}

    next_id = 256

    while next_id < vocab_size:
        pair_counts = {}

        # Count adjacent pairs, including overlapping occurrences
        for word in words:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        if not pair_counts:
            break

        # Most frequent pair; lexicographically greatest byte strings on ties
        best_pair = max(
            pair_counts,
            key=lambda p: (
                pair_counts[p],
                token_bytes[p[0]],
                token_bytes[p[1]]
            )
        )

        left, right = best_pair

        # Create the new token
        new_id = next_id
        next_id += 1

        merged_bytes = token_bytes[left] + token_bytes[right]
        token_bytes[new_id] = merged_bytes

        vocab.append([new_id, list(merged_bytes)])
        merges.append([left, right, new_id])

        # Replace matches from left to right without overlap
        new_words = []

        for word in words:
            new_word = []
            i = 0

            while i < len(word):
                if (
                    i + 1 < len(word)
                    and word[i] == left
                    and word[i + 1] == right
                ):
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_words.append(new_word)

        words = new_words

    return {
        "vocab": vocab,
        "merges": merges
    }
