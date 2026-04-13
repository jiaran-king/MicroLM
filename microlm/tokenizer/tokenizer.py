import regex as re
from collections.abc import Iterable
import json
from .bpe import bytes_to_unicode
'''
__init__:
传入初始参数 merge vocab special_tokens
构造 id_to_vocab vocab_to_id 以及merge_id词典
提取特殊字符

encoder:
根据特殊字符对语料进行切分
将切分的语料送入 _encode_text_segment中处理
处理special_token
返回ID列表

encode_text_segment:
gpt2预分词 将整段字符串切分成单个单词
对于每一个单词：
寻找best_pair进行合并

decoder:
id_to_vocab
合并
'''

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],  special_tokens:list[str]|None=None):
        self.id_to_vocab = dict(vocab)
        self.vocab_to_id = {v: id for id, v in self.id_to_vocab.items()}
        self.merges_id = {m: id for id, m in enumerate(merges)}
        self.special_tokens = special_tokens or []

        # Register special tokens into vocab with new IDs (starting after last BPE token)
        next_id = max(self.id_to_vocab.keys()) + 1 if self.id_to_vocab else 0
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in self.vocab_to_id:
                self.id_to_vocab[next_id] = tok_bytes
                self.vocab_to_id[tok_bytes] = next_id
                next_id += 1

        if self.special_tokens:
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            special_regex = "|".join(re.escape(t) for t in sorted_special)
            self.special_regex = re.compile(special_regex)
        else:
            self.special_regex = None
        self.gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

        if not raw_vocab:
            vocab_items: list[tuple[int, str]] = []
        elif all(isinstance(k, str) and k.isdigit() and isinstance(v, str) for k, v in raw_vocab.items()):
            vocab_items = [(int(k), v) for k, v in raw_vocab.items()]
        elif all(isinstance(k, str) and isinstance(v, int) for k, v in raw_vocab.items()):
            vocab_items = [(v, k) for k, v in raw_vocab.items()]
        else:
            raise ValueError("Unsupported vocab.json format")

        vocab = {
            token_id: bytes(byte_decoder[ch] for ch in token_text)
            for token_id, token_text in vocab_items
        }

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    merges.append(
                        (
                            bytes(byte_decoder[ch] for ch in parts[0]),
                            bytes(byte_decoder[ch] for ch in parts[1]),
                        )
                    )

        return cls(vocab, merges, special_tokens)

    def encode(self, text)->list[int]:
        if not text:
            return []
        if self.special_regex == None:
            return self._encode_text_segment(text)
        tokens = []
        last_pos = 0
        for match in self.special_regex.finditer(text):
            pre_text = text[last_pos:match.start()]
            if pre_text:
                tokens.extend(self._encode_text_segment(pre_text))
            special_token = match.group()
            special_token = self.vocab_to_id[special_token.encode("utf-8")]
            tokens.append(special_token)
            last_pos = match.end()
        if text[last_pos:]:
            tokens.extend(self._encode_text_segment(text[last_pos:]))
        return tokens

    def _encode_text_segment(self, text:str)-> list[int]:
        ids = []
        pre_tokens = self.gpt2_pat.findall(text)
        for word in pre_tokens:
            tokens = [bytes([b]) for b in word.encode("utf-8")]
            while(len(tokens) > 1):
                best_pair = None
                best_id = float("inf")
                for i in range(len(tokens)-1):
                    pair = (tokens[i], tokens[i+1])
                    if pair in self.merges_id and self.merges_id[pair] < best_id:
                        best_pair = pair
                        best_id = self.merges_id[pair]
                if best_pair == None:
                    break
                i = 0
                new_tokens = []
                while(i<len(tokens)):
                    if i<len(tokens) -1 and (tokens[i], tokens[i+1]) == best_pair:
                        new_tokens.append(best_pair[0] + best_pair[1])
                        i+=2
                    else:
                        new_tokens.append(tokens[i])
                        i+=1
                tokens = new_tokens
            for t in tokens:
                ids.append(self.vocab_to_id[t])
        return ids
    
    def decode(self, ids:list[int]) ->str:
        byte_segments = [self.id_to_vocab[i] for i in ids]
        full_bytes = b"".join(byte_segments)
        return full_bytes.decode("utf-8", errors="replace")
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            
            # 寻找安全的截断边界：优先寻找最后一个换行符，其次寻找最后一个空格
            # GPT-2 的正则天然会在换行符或空格处进行切分，因此在这里截断是绝对安全的
            safe_idx = max(buffer.rfind('\n'), buffer.rfind(' '))
            
            # 如果找到了安全边界
            if safe_idx != -1:
                # 截取从开头到安全边界（包含边界字符本身）的文本
                safe_text = buffer[:safe_idx + 1]
                # 对安全文本进行编码，并通过 yield from 逐个产出 token ID
                yield from self.encode(safe_text)
                
                # 将剩下未处理的“尾巴”重新赋值给 buffer，等待与下一个 chunk 拼接
                buffer = buffer[safe_idx + 1:]
                
        # 当整个 iterable 被遍历完后，如果 buffer 里还有残留的文本，进行最后一次编码
        if buffer:
            yield from self.encode(buffer)
