"""
╔══════════════════════════════════════════════════════════════╗
║            MULTI-HEAD CAUSAL SELF-ATTENTION                 ║
║                                                              ║
║  Серце кожного Transformer-а.                               ║
║                                                              ║
║  Self-Attention дозволяє кожному токену "дивитись" на всі   ║
║  попередні токени і вирішувати, на які з них звернути       ║
║  найбільшу увагу.                                           ║
║                                                              ║
║  Multi-Head: ми запускаємо КІЛЬКА механізмів уваги          ║
║  паралельно. Кожна "голова" вчиться знаходити різні         ║
║  зв'язки між токенами:                                      ║
║    - одна голова може вивчити синтаксис                      ║
║    - інша — семантичну близькість                            ║
║    - третя — позиційні залежності                            ║
║                                                              ║
║  Формула Attention:                                          ║
║    Attention(Q,K,V) = softmax(QK^T / √d_k) · V             ║
║                                                              ║
║  "Causal" означає що токен може бачити ТІЛЬКИ попередні     ║
║  токени (не майбутні) — це реалізується каузальною маскою.  ║
╚══════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.

    Замість окремих голів (як у старій версії), ми використовуємо
    ОДИН великий лінійний шар, який проектує одразу всі голови.
    Це ефективніше, бо GPU краще обробляє одну велику матрицю
    ніж багато маленьких.

    Args:
        n_embed:    Розмірність ембедингу (наприклад, 128)
        n_heads:    Кількість голів уваги (наприклад, 4)
        block_size: Максимальна довжина послідовності (контексту)
        dropout:    Ймовірність Dropout (0.0 — 1.0)
    """

    def __init__(self, n_embed, n_heads, block_size, dropout=0.1):
        super().__init__()

        # Перевіряємо що n_embed ділиться на n_heads без залишку
        # Бо кожна голова отримує n_embed // n_heads розмірність
        assert n_embed % n_heads == 0, \
            f"n_embed ({n_embed}) має ділитися на n_heads ({n_heads})"

        self.n_heads = n_heads
        self.n_embed = n_embed
        self.head_size = n_embed // n_heads  # Розмір кожної голови

        # ─── Проекції Q, K, V ───
        # Один лінійний шар проектує вхід у Q, K, V одразу для ВСІХ голів.
        # Це ефективніше ніж n_heads окремих лінійних шарів.
        #
        # Query (Q) — "що я шукаю?" — кожен токен формулює запит
        # Key (K)   — "що я пропоную?" — кожен токен оголошує себе
        # Value (V) — "яка моя інформація?" — реальний контент токена
        #
        # Розмір: (n_embed) → (3 * n_embed), потім розділимо на Q, K, V
        self.qkv_proj = nn.Linear(n_embed, 3 * n_embed, bias=False)

        # ─── Вихідна проекція ───
        # Після конкатенації всіх голів, проектуємо назад у n_embed
        # Це дозволяє моделі "перемішати" інформацію з різних голів
        self.out_proj = nn.Linear(n_embed, n_embed, bias=False)

        # ─── Dropout ───
        # Випадково "вимикає" частину зв'язків уваги під час тренування.
        # Це змушує модель не покладатися на конкретні зв'язки,
        # і робить її більш робастною (стійкою).
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # ─── Каузальна маска ───
        # Нижньо-трикутна матриця розміром (block_size × block_size).
        # Вона забороняє кожному токену "підглядати" у майбутнє:
        #
        #   [1, 0, 0, 0]    ← токен 1 бачить лише себе
        #   [1, 1, 0, 0]    ← токен 2 бачить 1 і себе
        #   [1, 1, 1, 0]    ← токен 3 бачить 1, 2 і себе
        #   [1, 1, 1, 1]    ← токен 4 бачить усіх
        #
        # register_buffer — зберігає тензор як частину моделі,
        # але НЕ як параметр (не оновлюється при навчанні)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
                 .view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        """
        Прямий прохід Multi-Head Attention.

        Вхід:  x — тензор (B, T, C) де:
               B = batch size (кількість прикладів)
               T = sequence length (довжина послідовності)
               C = n_embed (розмірність ембедингу)

        Вихід: тензор (B, T, C) — такого ж розміру
        """
        B, T, C = x.shape

        # ─── Крок 1: Проекція Q, K, V ───
        # Один матричний множення замість трьох
        # qkv має розмір (B, T, 3*C)
        qkv = self.qkv_proj(x)

        # Розділяємо на Q, K, V — кожен розміром (B, T, C)
        q, k, v = qkv.chunk(3, dim=-1)

        # ─── Крок 2: Розділяємо на голови ───
        # Перетворюємо (B, T, C) → (B, n_heads, T, head_size)
        # Це дозволяє обробити всі голови паралельно
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_size).transpose(1, 2)

        # ─── Крок 3: Scaled Dot-Product Attention ───
        # Використовуємо вбудовану оптимізовану функцію PyTorch
        # Вона робить те саме: softmax(QK^T / √d_k) · V + каузальна маска
        # Але ЗНАЧНО швидше завдяки оптимізаціям (FlashAttention, memory-efficient)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,  # автоматична каузальна маска
        )

        # ─── Крок 5: Конкатенація голів ───
        # Повертаємо (B, n_heads, T, head_size) → (B, T, C)
        # contiguous() потрібен після transpose для коректного view
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # ─── Крок 6: Вихідна проекція + Dropout ───
        # Фінальне лінійне перетворення "змішує" інформацію з усіх голів
        out = self.resid_dropout(self.out_proj(out))

        return out
