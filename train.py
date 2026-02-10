"""
╔══════════════════════════════════════════════════════════════╗
║                  ТРЕНУВАЛЬНИЙ PIPELINE                      ║
║                                                              ║
║  Повний цикл тренування LLM моделі:                        ║
║                                                              ║
║  1. Завантаження та підготовка даних                        ║
║  2. Навчання BPE токенізатора                               ║
║  3. Train/Val split (90%/10%)                               ║
║  4. Тренувальний цикл з:                                    ║
║     - Cosine Annealing LR scheduler з warmup               ║
║     - Gradient clipping                                      ║
║     - Періодична валідація                                  ║
║     - Checkpointing (збереження моделі)                     ║
║     - GPU підтримка (CUDA)                                  ║
║  5. Фінальна генерація для перевірки                        ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import math
import time
import torch
import torch.nn.functional as F

from tokenizer import BPETokenizer
from model import GPT, GPTConfig


# ═══════════════════════════════════════════
# КОНФІГУРАЦІЯ ТРЕНУВАННЯ
# ═══════════════════════════════════════════

# ─── Дані ───
DATA_FILE = "data.txt"
CHECKPOINT_DIR = "checkpoints"

# ─── Токенізатор ───
VOCAB_SIZE = 500          # Розмір BPE словника (менший = швидше)

# ─── Модель ───
BLOCK_SIZE = 128          # Максимальна довжина контексту
                          # 128 замість 256 → attention в 4x швидше (O(T²))
N_EMBED = 128             # Розмірність ембедингу
N_HEADS = 4               # Кількість голів уваги
N_LAYERS = 4              # Кількість Transformer блоків
DROPOUT = 0.1             # Dropout rate

# ─── Тренування ───
BATCH_SIZE = 16           # Менший batch = швидше на CPU
LEARNING_RATE = 5e-4      # Швидкість навчання
MAX_STEPS = 5000          # Кількість кроків (5000 замість 10000)
WARMUP_STEPS = 200        # Warmup кроки
EVAL_INTERVAL = 500       # Оцінка кожні N кроків
EVAL_ITERS = 15           # Скільки batch-ів при оцінці
SAVE_INTERVAL = 1000      # Збереження кожні N кроків
GRAD_CLIP = 1.0           # Gradient clipping
WEIGHT_DECAY = 0.01       # L2-регуляризація

# ─── Пристрій ───
# CUDA = GPU від NVIDIA, MPS = GPU на Apple Silicon Mac
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ═══════════════════════════════════════════
# LEARNING RATE SCHEDULER
# ═══════════════════════════════════════════

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=1e-5):
    """
    Cosine Annealing з Linear Warmup.

    Це стандартний LR scheduler для тренування LLM:

    1. Warmup (0 → warmup_steps):
       LR лінійно зростає від 0 до max_lr.
       Чому: На початку ваги випадкові і великий LR може
       "зламати" модель. Поступове збільшення дає моделі
       час "адаптуватися".

    2. Cosine Decay (warmup_steps → max_steps):
       LR плавно знижується за косинусом від max_lr до min_lr.
       Чому: На початку модель вчить грубі паттерни (потрібен
       високий LR), потім — тонкі деталі (потрібен низький LR).
       Косинус плавніший за лінійний decay.

    Графічно:
       LR │     ╱‾‾‾‾‾╲
          │    ╱        ╲
          │   ╱          ╲
          │  ╱            ╲_____
          │_╱
          └──────────────────── step
          0   warmup    max_steps
    """
    # Warmup: лінійне зростання
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)

    # Після max_steps: мінімальний LR
    if step >= max_steps:
        return min_lr

    # Cosine decay
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ═══════════════════════════════════════════
# ОТРИМАННЯ BATCH ДАНИХ
# ═══════════════════════════════════════════

def get_batch(data, block_size, batch_size, device):
    """
    Формує випадковий batch для тренування.

    З масиву токенів вирізаємо випадкові "вікна" довжиною block_size.

    Приклад (block_size=4):
      data = [10, 20, 30, 40, 50, 60, 70]

      Випадкова позиція i=2:
        x = [30, 40, 50, 60]   ← вхід (контекст)
        y = [40, 50, 60, 70]   ← ціль (зсунута на 1 вправо)

      Модель вчиться:
        бачу [30] → передбачаю 40
        бачу [30, 40] → передбачаю 50
        бачу [30, 40, 50] → передбачаю 60
        бачу [30, 40, 50, 60] → передбачаю 70

    Args:
        data:       1D тензор усіх токенів
        block_size: Довжина контексту
        batch_size: Кількість прикладів у batch
        device:     Пристрій (cpu/cuda)
    """
    # Випадкові стартові позиції
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Формуємо вхід і ціль
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    # Переміщуємо на потрібний пристрій (GPU/CPU)
    return x.to(device), y.to(device)


# ═══════════════════════════════════════════
# ОЦІНКА LOSS
# ═══════════════════════════════════════════

@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_iters):
    """
    Оцінює середній loss на тренувальних і валідаційних даних.

    @torch.no_grad() — вимикає обчислення градієнтів.
    Під час оцінки нам не потрібні градієнти, і це:
    - Економить пам'ять GPU
    - Прискорює обчислення

    Ми усереднюємо loss по eval_iters batch-ах для стабільнішої оцінки.
    Один batch може дати "шумний" результат.
    """
    model.eval()   # Вимикаємо Dropout
    out = {}

    for split_name, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(data, block_size, batch_size, device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split_name] = sum(losses) / len(losses)

    model.train()  # Повертаємо Dropout
    return out


# ═══════════════════════════════════════════
# ЗБЕРЕЖЕННЯ / ЗАВАНТАЖЕННЯ CHECKPOINT
# ═══════════════════════════════════════════

def save_checkpoint(model, optimizer, step, loss, config, path):
    """
    Зберігає повний стан тренування.

    Зберігаємо:
    - model_state_dict: ваги моделі
    - optimizer_state_dict: стан оптимізатора (momentum, etc.)
    - step: поточний крок (щоб продовжити з того ж місця)
    - loss: поточний loss
    - config: конфігурація моделі (для відтворення архітектури)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "loss": loss,
        "config": config.__dict__,
    }, path)
    print(f"💾 Checkpoint збережено: {path}")


# ═══════════════════════════════════════════
# ГОЛОВНА ФУНКЦІЯ ТРЕНУВАННЯ
# ═══════════════════════════════════════════

def train():
    print("=" * 60)
    print("        🚀 ТРЕНУВАННЯ LLM МОДЕЛІ")
    print("=" * 60)
    print(f"📱 Пристрій: {DEVICE}")

    # ─── 1. Завантаження даних ───
    print("\n📄 Завантаження даних...")
    text = open(DATA_FILE, encoding="utf-8").read()
    print(f"   Символів: {len(text):,}")

    # ─── 2. Навчання токенізатора ───
    print()
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=VOCAB_SIZE)

    # Токенізуємо весь текст
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"   Токенів:  {len(data):,}")

    # ─── 3. Train/Val Split ───
    # 90% — тренувальні дані
    # 10% — валідаційні (для контролю перенавчання)
    #
    # Чому розділяємо: якщо оцінювати якість на тих самих даних
    # що й тренуємо, ми не побачимо перенавчання. Val loss
    # показує наскільки модель "розуміє" нові дані, а не запам'ятовує.
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"\n📊 Split: train={len(train_data):,} | val={len(val_data):,}")

    # ─── 4. Створення моделі ───
    print()
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_embed=N_EMBED,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    )
    model = GPT(config).to(DEVICE)

    # torch.compile вимкнений (потребує C++ компілятор на Windows)

    # ─── 5. Оптимізатор AdamW ───
    # AdamW = Adam + правильний Weight Decay
    #
    # Adam — адаптивний оптимізатор, який підлаштовує learning rate
    # для кожного параметра окремо на основі:
    # - Momentum (ковзне середнє градієнтів) — згладжує напрям
    # - RMSprop (ковзне середнє квадратів градієнтів) — адаптує крок
    #
    # Weight Decay (L2 reg) — штрафує великі ваги, запобігаючи
    # перенавчанню. AdamW робить це ПРАВИЛЬНО (на відміну від Adam).
    #
    # Ми НЕ застосовуємо weight decay до bias та LayerNorm,
    # бо вони не схильні до перенавчання.
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # bias та LayerNorm параметри — без weight decay
        if "bias" in name or "ln" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=LEARNING_RATE, betas=(0.9, 0.95))
    # betas=(0.9, 0.95) — стандарт для тренування LLM
    # beta1=0.9: momentum (90% попередніх градієнтів)
    # beta2=0.95: RMSprop (95% попередніх квадратів)

    # ─── 6. Тренувальний цикл ───
    print(f"\n🏋️ Тренування ({MAX_STEPS} кроків)...")
    print("-" * 60)

    best_val_loss = float("inf")
    start_time = time.time()

    for step in range(MAX_STEPS):
        # Оновлюємо learning rate
        lr = get_lr(step, WARMUP_STEPS, MAX_STEPS, LEARNING_RATE)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Отримуємо batch
        xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)

        # Forward pass
        logits, loss = model(xb, yb)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)  # Очищуємо градієнти
        loss.backward()                         # Обчислюємо нові градієнти

        # Gradient Clipping — обмежуємо норму градієнтів
        # Якщо градієнт занадто великий, він пропорційно зменшується.
        # Це запобігає "вибуху градієнтів" у глибоких мережах.
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()                        # Оновлюємо ваги

        # ─── Оцінка ───
        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
            losses = estimate_loss(
                model, train_data, val_data,
                BLOCK_SIZE, BATCH_SIZE, DEVICE, EVAL_ITERS
            )
            elapsed = time.time() - start_time

            # Зберігаємо найкращу модель
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(
                    model, optimizer, step, losses["val"],
                    config, f"{CHECKPOINT_DIR}/best_model.pt"
                )

            print(
                f"  Step {step:5d}/{MAX_STEPS} | "
                f"Train Loss: {losses['train']:.4f} | "
                f"Val Loss: {losses['val']:.4f} | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

        # ─── Periodic Checkpoint ───
        if step > 0 and step % SAVE_INTERVAL == 0:
            save_checkpoint(
                model, optimizer, step, loss.item(),
                config, f"{CHECKPOINT_DIR}/step_{step}.pt"
            )

    # ─── 7. Фінальне збереження ───
    save_checkpoint(
        model, optimizer, MAX_STEPS, loss.item(),
        config, f"{CHECKPOINT_DIR}/final_model.pt"
    )
    tokenizer.save(f"{CHECKPOINT_DIR}/tokenizer.json")

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"✅ Тренування завершено за {total_time:.1f}с")
    print(f"   Найкращий val loss: {best_val_loss:.4f}")

    # ─── 8. Тестова генерація ───
    print("\n📝 Тестова генерація:")
    print("-" * 40)

    # Починаємо з BOS токена або нульового
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    generated = model.generate(
        context, max_new_tokens=200,
        temperature=0.8, top_k=50
    )
    print(tokenizer.decode(generated[0].tolist()))
    print("-" * 40)

    return model, tokenizer


# ═══════════════════════════════════════════
# ТОЧКА ВХОДУ
# ═══════════════════════════════════════════

if __name__ == "__main__":
    train()
