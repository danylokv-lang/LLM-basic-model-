"""
╔══════════════════════════════════════════════════════════════╗
║                 ІНТЕРАКТИВНИЙ ЧАТ З LLM                    ║
║                                                              ║
║  Завантажує натреновану модель і дозволяє генерувати текст  ║
║  в інтерактивному режимі.                                   ║
║                                                              ║
║  Параметри генерації:                                       ║
║    /temp <value>   — змінити temperature                     ║
║    /topk <value>   — змінити top-k                           ║
║    /topp <value>   — змінити top-p                           ║
║    /len <value>    — змінити довжину генерації                ║
║    /help           — показати допомогу                        ║
║    /quit           — вийти                                    ║
╚══════════════════════════════════════════════════════════════╝
"""

import torch
from tokenizer import BPETokenizer
from model import GPT, GPTConfig


def load_model(checkpoint_path, tokenizer_path, device="cpu"):
    """
    Завантажує натреновану модель з checkpoint-у.

    Процес:
    1. Завантажуємо токенізатор (потрібен для encode/decode)
    2. Зчитуємо checkpoint (ваги + конфіг)
    3. Відтворюємо архітектуру моделі з конфігу
    4. Завантажуємо ваги
    5. Переводимо в eval режим (вимикаємо Dropout)
    """
    # Завантажуємо токенізатор
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)

    # Завантажуємо checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Відтворюємо конфіг і модель
    config = GPTConfig(**checkpoint["config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()  # Evaluation mode — Dropout вимкнений

    step = checkpoint.get("step", "?")
    loss = checkpoint.get("loss", "?")
    print(f"✅ Модель завантажено (step={step}, loss={loss:.4f})")

    return model, tokenizer, config


def generate_text(model, tokenizer, prompt, device,
                  max_tokens=200, temperature=0.8, top_k=50, top_p=None):
    """
    Генерує продовження тексту на основі prompt-у.

    Args:
        prompt:      Початковий текст (або порожній рядок)
        max_tokens:  Кількість нових токенів
        temperature: Контроль креативності (0.1—2.0)
        top_k:       Топ-K фільтрація
        top_p:       Nucleus sampling
    """
    if prompt:
        # Кодуємо prompt у токени
        token_ids = tokenizer.encode(prompt)
        idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    else:
        # Починаємо з нульового токена
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    # Генеруємо
    with torch.no_grad():
        output = model.generate(
            idx,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    # Декодуємо назад у текст
    return tokenizer.decode(output[0].tolist())


def print_help():
    """Показує довідку по командам."""
    print("""
╔════════════════════════════════════════╗
║            КОМАНДИ                    ║
╠════════════════════════════════════════╣
║  /temp <0.1-2.0>  Temperature         ║
║  /topk <1-100>    Top-K sampling      ║
║  /topp <0.1-1.0>  Top-P (nucleus)     ║
║  /len <10-500>    Довжина генерації    ║
║  /help            Ця довідка          ║
║  /quit            Вийти               ║
╚════════════════════════════════════════╝
    """)


def chat():
    """Головний цикл інтерактивного чату."""

    print("=" * 50)
    print("     LLM ІНТЕРАКТИВНИЙ ЧАТ")
    print("=" * 50)

    # Вибір пристрою
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f" Пристрій: {device}")

    # Завантаження моделі
    checkpoint_path = "checkpoints/best_model.pt"
    tokenizer_path = "checkpoints/tokenizer.json"

    try:
        model, tokenizer, config = load_model(
            checkpoint_path, tokenizer_path, device
        )
    except FileNotFoundError:
        print(" Checkpoint не знайдено!")
        print(f"   Спочатку натренуйте модель: python train.py")
        print(f"   Очікується: {checkpoint_path}")
        return

    # Параметри генерації (можна змінювати командами)
    params = {
        "temperature": 0.8,
        "top_k": 50,
        "top_p": None,
        "max_tokens": 200,
    }

    print_help()
    print(f"Параметри: temp={params['temperature']}, "
          f"top_k={params['top_k']}, "
          f"top_p={params['top_p']}, "
          f"len={params['max_tokens']}")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nВи: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n До побачення!")
            break

        if not user_input:
            continue

        # ─── Обробка команд ───
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd == "/quit":
                print(" До побачення!")
                break
            elif cmd == "/help":
                print_help()
            elif cmd == "/temp" and len(parts) == 2:
                try:
                    params["temperature"] = float(parts[1])
                    print(f"   Temperature = {params['temperature']}")
                except ValueError:
                    print("   Невірне значення")
            elif cmd == "/topk" and len(parts) == 2:
                try:
                    params["top_k"] = int(parts[1])
                    print(f"   Top-K = {params['top_k']}")
                except ValueError:
                    print("   Невірне значення")
            elif cmd == "/topp" and len(parts) == 2:
                try:
                    val = float(parts[1])
                    params["top_p"] = val if val < 1.0 else None
                    print(f"   Top-P = {params['top_p']}")
                except ValueError:
                    print("   Невірне значення")
            elif cmd == "/len" and len(parts) == 2:
                try:
                    params["max_tokens"] = int(parts[1])
                    print(f"   Max tokens = {params['max_tokens']}")
                except ValueError:
                    print("   Невірне значення")
            else:
                print("  Невідома команда. /help для довідки")
            continue

        # ─── Генерація ───
        print("\n Модель:", end=" ", flush=True)
        output = generate_text(
            model, tokenizer, user_input, device,
            max_tokens=params["max_tokens"],
            temperature=params["temperature"],
            top_k=params["top_k"],
            top_p=params["top_p"],
        )
        print(output)


if __name__ == "__main__":
    chat()
