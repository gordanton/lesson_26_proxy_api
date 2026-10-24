"""Диалог через Anthropic Messages API и ProxyAPI.

Каркас взят из шаблона документации: клиент Anthropic, модель claude-sonnet-4-5,
режим thinking и адрес прокси. Память устроена как в chat.py: в каждый запрос
уходит вся история, поэтому чат не обрывается после одного ответа.
"""

import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

# У OpenAI в адресе есть /v1, у Anthropic его дописывает сам SDK.
PROXY_BASE_URL = "https://api.proxyapi.ru/anthropic"
MODEL = "claude-sonnet-4-5"

# В слайде стоит budget_tokens=500. Текущий API такое значение отклоняет:
# бюджет thinking должен быть не меньше 1024 и строго меньше max_tokens.
# max_tokens — это общий потолок: размышление плюс видимый ответ.
THINKING = {"type": "enabled", "budget_tokens": 1024}
MAX_TOKENS = 2048
SYSTEM_PROMPT = "Ты дружелюбный помощник. Отвечай кратко, по-русски."


def make_client() -> anthropic.Anthropic:
    """Клиент Anthropic через ProxyAPI. Ключ читается из .env."""
    # В шаблоне документации переменная называется ANTHROPIC_API_KEY.
    # У ProxyAPI ключ один на все модели, поэтому подойдёт и уже сохранённый OPENAI_API_KEY.
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "В .env нет ключа. Добавьте ANTHROPIC_API_KEY или OPENAI_API_KEY."
        )

    return anthropic.Anthropic(api_key=api_key, base_url=PROXY_BASE_URL)


def new_dialog() -> list[dict]:
    """Пустая история. Роль ассистента у Anthropic задаётся параметром system, не сообщением."""
    return []


def visible_text(content) -> str:
    """Оставляет только текст ответа. Блоки thinking пользователю не показываем."""
    parts = []
    for block in content:
        block_type = block["type"] if isinstance(block, dict) else block.type
        if block_type == "text":
            parts.append(block["text"] if isinstance(block, dict) else block.text)
    return "\n".join(parts)


def generate_reply(client: anthropic.Anthropic, messages: list[dict], user_text: str) -> str:
    """Один ход диалога: сохраняет реплику, запрашивает ответ и кладёт его в историю целиком."""
    messages.append(
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        }
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        thinking=THINKING,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    # В историю нужно вернуть и thinking, и текст, вместе с полем signature.
    # Если оставить только текст, следующий запрос с thinking завершится ошибкой.
    messages.append(
        {
            "role": "assistant",
            "content": [block.model_dump(exclude_none=True) for block in response.content],
        }
    )
    return visible_text(response.content)


def chat_loop(client: anthropic.Anthropic, messages: list[dict]) -> None:
    """Крутит диалог, пока пользователь сам его не закроет."""
    print("Диалог запущен. Чтобы выйти, напишите /exit или оставьте строку пустой.")
    while True:
        user_text = input("Вы: ").strip()
        if not user_text or user_text.lower() in {"/exit", "exit", "выход"}:
            print("Чат завершён.")
            break

        answer = generate_reply(client, messages, user_text)
        print(f"Ассистент: {answer}\n")


if __name__ == "__main__":
    client = make_client()
    messages = new_dialog()
    chat_loop(client, messages)
