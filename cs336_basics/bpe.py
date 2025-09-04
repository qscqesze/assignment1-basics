import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple


class BPE:
    def __init__(self, vocab_size: int = 1000, encoding: str = "utf-8"):
        self.target_vocab_size = vocab_size
        self.encoding_name = encoding
        self.vocab = {}
        self.token_to_id = {}
        self.merges = []
        self.trained = False

    def _get_stats(self, words: Dict[Tuple[bytes, ...], int]) -> Counter:
        pairs = Counter()
        for word, freq in words.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i+1])] += freq
        return pairs

    def _merge_vocab(self, pair: Tuple[bytes, bytes], words: Dict[Tuple[bytes, ...], int]) -> Dict[Tuple[bytes, ...], int]:
        new_words = {}
        bigram = pair[0] + pair[1]
        
        for word in words:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                    new_word.append(bigram)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words[tuple(new_word)] = words[word]
        return new_words

    def train(self, input_path: str, vocab_size: int, special_tokens: list[str]) -> dict[int, bytes]:
        with open(input_path, 'r', encoding=self.encoding_name) as f:
            text = f.read()
        
        text_bytes = text.encode(self.encoding_name)
        
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.token_to_id = {bytes([i]): i for i in range(256)}
        
        next_id = 256
        for special_token in special_tokens:
            special_bytes = special_token.encode(self.encoding_name)
            if special_bytes not in self.token_to_id:
                self.vocab[next_id] = special_bytes
                self.token_to_id[special_bytes] = next_id
                next_id += 1
        
        words = {}
        for word in text.split():
            word_bytes = word.encode(self.encoding_name)
            word_tuple = tuple(bytes([b]) for b in word_bytes)
            words[word_tuple] = words.get(word_tuple, 0) + 1
        
        current_vocab_size = len(self.vocab)
        while current_vocab_size < vocab_size:
            pairs = self._get_stats(words)
            if not pairs:
                break
            
            best_pair = pairs.most_common(1)[0][0]
            words = self._merge_vocab(best_pair, words)
            
            new_token = best_pair[0] + best_pair[1]
            self.vocab[current_vocab_size] = new_token
            self.token_to_id[new_token] = current_vocab_size
            self.merges.append(best_pair)
            current_vocab_size += 1
        
        self.trained = True
        return self.vocab

    def encode(self, text: str) -> list[int]:
        if not self.trained:
            raise ValueError("Model not trained")
        
        text_bytes = text.encode(self.encoding_name)
        tokens = [bytes([b]) for b in text_bytes]
        
        for pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                    merged_token = pair[0] + pair[1]
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return [self.token_to_id[token] for token in tokens]

    def decode(self, indices: list[int]) -> str:
        if not self.trained:
            raise ValueError("Model not trained")
        
        byte_sequence = b''.join([self.vocab[idx] for idx in indices])
        return byte_sequence.decode(self.encoding_name)


if __name__ == "__main__":
    sample_text = """
    Hello world! This is a sample text for training the BPE model.
    The quick brown fox jumps over the lazy dog.
    BPE (Byte Pair Encoding) is a data compression technique.
    """
    
    with open("sample_text.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    bpe = BPE(vocab_size=300, encoding="utf-8")
    vocab = bpe.train("sample_text.txt", vocab_size=300, special_tokens=["<PAD>", "<UNK>"])
    print(vocab)
    test_text = "Hello world!"
    encoded = bpe.encode(test_text)
    decoded = bpe.decode(encoded)
    
    print(f"Text: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Vocab size: {len(vocab)}")
    
    os.remove("sample_text.txt")