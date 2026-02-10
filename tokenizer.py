"""
╔══════════════════════════════════════════════════════════════╗
║                   BPE ТОКЕНІЗАТОР                           ║
║                                                              ║
║  Byte-Pair Encoding — стандартний алгоритм токенізації       ║
║  у GPT, LLaMA та інших LLM моделях.                        ║
║                                                              ║
║  Як працює BPE:                                              ║
║  1. Розбиваємо текст на окремі символи (початковий словник) ║
║  2. Знаходимо найчастішу пару сусідніх токенів              ║
║  3. Зливаємо цю пару в один новий токен                     ║
║  4. Повторюємо кроки 2-3 поки словник не досягне потрібного ║
║     розміру                                                  ║
║                                                              ║
║  Приклад: "ааб ааб" → 'а'+'а'='аа' → 'аа'+'б'='ааб'       ║
╚══════════════════════════════════════════════════════════════╝
"""

import json
import os


class BPETokenizer:
    """
    Byte-Pair Encoding токенізатор.

    Спеціальні токени:
      <PAD> = 0  — padding (заповнення до однакової довжини)
      <UNK> = 1  — невідомий токен
      <BOS> = 2  — початок послідовності (Begin Of Sequence)
      <EOS> = 3  — кінець послідовності (End Of Sequence)
    """

    def __init__(self):
        # ─── Спеціальні токени ───
        # PAD - вирівнює послідовності до однієї довжини в batch
        # UNK - замінює символи яких немає у словнику
        # BOS - сигнал моделі що починається новий текст
        # EOS - сигнал моделі що текст закінчився
        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
        }
        self.num_special = len(self.special_tokens)

        # Словники для перетворення токен ↔ індекс
        self.token_to_id = {}   # "при" → 42
        self.id_to_token = {}   # 42 → "при"

        # Список злиттів BPE у порядку навчання
        # Кожне злиття = (токен_A, токен_B) → новий_токен
        self.merges = []

        self.vocab_size = 0

    def _get_pairs(self, tokens):
        """
        Знаходить усі пари сусідніх токенів та рахує їхню частоту.

        Наприклад, для ['п', 'р', 'и', 'в', 'і', 'т']:
        пари: {('п','р'): 1, ('р','и'): 1, ('и','в'): 1, ...}

        Це ядро BPE — ми завжди зливаємо найчастішу пару.
        """
        pairs = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] = pairs.get(pair, 0) + 1
        return pairs

    def _merge_pair(self, tokens, pair):
        """
        Зливає всі входження пари (A, B) → AB у послідовності.

        Наприклад:
          tokens = ['п', 'р', 'и', 'в', 'і', 'т']
          pair = ('п', 'р')
          результат = ['пр', 'и', 'в', 'і', 'т']
        """
        new_tokens = []
        i = 0
        merged = pair[0] + pair[1]

        while i < len(tokens):
            if (i < len(tokens) - 1
                    and tokens[i] == pair[0]
                    and tokens[i + 1] == pair[1]):
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        return new_tokens

    def train(self, text, vocab_size=500):
        """
        Навчає BPE словник на тексті.

        Алгоритм:
        1. Починаємо з алфавіту окремих символів
        2. Повторюємо:
           a) Рахуємо частоту кожної пари сусідніх токенів
           b) Знаходимо найчастішу пару
           c) Зливаємо цю пару у новий токен
           d) Додаємо злиття у список merges
        3. Зупиняємося коли досягли потрібного vocab_size
        """
        print(f"🔤 Навчання BPE токенізатора (vocab_size={vocab_size})...")

        # ── Крок 1: Початковий словник з окремих символів ──
        chars = sorted(set(text))
        self.token_to_id = dict(self.special_tokens)
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}

        for ch in chars:
            idx = len(self.token_to_id)
            self.token_to_id[ch] = idx
            self.id_to_token[idx] = ch

        # ── Крок 2: Розбиваємо текст на символи ──
        tokens = list(text)

        # ── Крок 3: Ітеративно зливаємо найчастіші пари ──
        self.merges = []
        num_merges = vocab_size - len(self.token_to_id)

        for i in range(num_merges):
            pairs = self._get_pairs(tokens)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            merged_token = best_pair[0] + best_pair[1]
            tokens = self._merge_pair(tokens, best_pair)

            idx = len(self.token_to_id)
            self.token_to_id[merged_token] = idx
            self.id_to_token[idx] = merged_token
            self.merges.append(best_pair)

            if (i + 1) % 100 == 0:
                print(f"  Злиття {i+1}/{num_merges}: "
                      f"'{best_pair[0]}' + '{best_pair[1]}' → '{merged_token}'")

        self.vocab_size = len(self.token_to_id)
        print(f"✅ Словник готовий: {self.vocab_size} токенів")

    def encode(self, text):
        """
        Перетворює текст у послідовність ID токенів.

        Процес:
        1. Розбиваємо текст на символи
        2. Послідовно застосовуємо ВСІ злиття у тому ж порядку
           як при навчанні — це гарантує детермінованість
        3. Кожен токен → його ID
        """
        tokens = list(text)

        for pair in self.merges:
            tokens = self._merge_pair(tokens, pair)

        unk_id = self.special_tokens["<UNK>"]
        return [self.token_to_id.get(t, unk_id) for t in tokens]

    def decode(self, ids):
        """
        Відновлює текст з послідовності ID.
        Пропускає спеціальні токени.
        """
        tokens = []
        for idx in ids:
            token = self.id_to_token.get(idx, "")
            if token in self.special_tokens:
                continue
            tokens.append(token)
        return "".join(tokens)

    def save(self, path):
        """Зберігає токенізатор у JSON файл."""
        data = {
            "token_to_id": self.token_to_id,
            "merges": self.merges,
            "vocab_size": self.vocab_size,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 Токенізатор збережено: {path}")

    def load(self, path):
        """Завантажує токенізатор з JSON файлу."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.token_to_id = data["token_to_id"]
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
        self.merges = [tuple(m) for m in data["merges"]]
        self.vocab_size = data["vocab_size"]
        print(f"📂 Токенізатор завантажено: {self.vocab_size} токенів")


# ═══════════════════════════════════════════
# Тестування
# ═══════════════════════════════════════════
if __name__ == "__main__":
    text = open("data.txt", encoding="utf-8").read()
    tok = BPETokenizer()
    tok.train(text, vocab_size=200)

    sample = "привіт світ"
    encoded = tok.encode(sample)
    decoded = tok.decode(encoded)

    print(f"\nТест:")
    print(f"  Вхід:      '{sample}'")
    print(f"  Encoded:   {encoded}")
    print(f"  Decoded:   '{decoded}'")
    print(f"  Збігається: {sample == decoded}")
