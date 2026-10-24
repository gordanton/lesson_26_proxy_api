"""Диалог через ProxyAPI с выбором модели при запуске.

По умолчанию работает думающая модель Claude (Anthropic).
Вторая клавиша переключает на обычную модель OpenAI.
История в обоих режимах хранится в списке messages и уходит в каждый запрос целиком.
"""

import os

import anthropic
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_PROXY_URL = "https://api.proxyapi.ru/openai/v1"
ANTHROPIC_PROXY_URL = "https://api.proxyapi.ru/anthropic"
CLAUDE_MODEL = "claude-sonnet-4-5"
# Бюджет thinking не может быть меньше 1024 и должен быть меньше max_tokens.
CLAUDE_THINKING = {"type": "enabled", "budget_tokens": 1024}
CLAUDE_MAX_TOKENS = 2048
SYSTEM_PROMPT = "Ты дружелюбный помощник. Отвечай кратко, по-русски."


def read_api_key() -> str:
    """Один ключ ProxyAPI подходит и Claude, и OpenAI."""
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "В .env нет ключа. Добавьте ANTHROPIC_API_KEY или OPENAI_API_KEY."
        )
    return api_key


def choose_mode() -> str:
    """Спрашивает режим один раз в начале. Пустой ввод оставляет думающую модель."""
    print("Выберите модель.")
    print("  Enter — думающая, Claude от Anthropic")
    print("  2     — обычная, OpenAI")
    choice = input("Режим: ").strip().lower()
    if choice in {"2", "обычная", "openai"}:
        return "openai"
    return "claude"


def make_openai_client() -> OpenAI:
    return OpenAI(api_key=read_api_key(), base_url=OPENAI_PROXY_URL)


def new_openai_dialog() -> list[dict]:
    """У OpenAI роль ассистента задаётся первым сообщением в истории."""
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def generate_openai_reply(client: OpenAI, messages: list[dict], user_text: str) -> str:
    """Один ход обычной модели. В историю кладётся только текст ответа."""
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    messages.append({"role": "user", "content": user_text})
    response = client.chat.completions.create(model=model, messages=messages)
    answer = response.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": answer})
    return answer


def make_claude_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=read_api_key(), base_url=ANTHROPIC_PROXY_URL)


def new_claude_dialog() -> list[dict]:
    """У Claude системная роль передаётся отдельно, поэтому история начинается пустой."""
    return []


def format_claude_message(response) -> str:
    """Показывает размышление и ответ без служебных полей вроде signature."""
    thinking_parts = []
    answer_parts = []
    for block in response.content:
        if block.type == "thinking" and block.thinking.strip():
            thinking_parts.append(block.thinking.strip())
        elif block.type == "text" and block.text.strip():
            answer_parts.append(block.text.strip())

    lines = [f"Модель: {response.model}", ""]
    if thinking_parts:
        lines.append("Размышление:")
        lines.append("\n\n".join(thinking_parts))
        lines.append("")
    lines.append("Ответ:")
    lines.append("\n".join(answer_parts) if answer_parts else "(модель не вернула текст)")
    lines.append("")
    lines.append(
        "Токены: "
        f"вход {response.usage.input_tokens}, выход {response.usage.output_tokens}"
    )
    return "\n".join(lines)


def generate_claude_reply(client: anthropic.Anthropic, messages: list[dict], user_text: str) -> str:
    """Один ход думающей модели. В историю возвращается весь ответ, включая thinking."""
    messages.append(
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        }
    )
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_MAX_TOKENS,
        thinking=CLAUDE_THINKING,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    # signature внутри блока thinking трогать нельзя: без него следующий ход сломается.
    messages.append(
        {
            "role": "assistant",
            "content": [block.model_dump(exclude_none=True) for block in response.content],
        }
    )
    return format_claude_message(response)


def chat_loop(mode: str, client, messages: list[dict]) -> None:
    """Крутит диалог в уже выбранном режиме, пока пользователь сам его не закроет."""
    print("Диалог запущен. Чтобы выйти, напишите /exit или оставьте строку пустой.")
    while True:
        user_text = input("Вы: ").strip()
        if not user_text or user_text.lower() in {"/exit", "exit", "выход"}:
            print("Чат завершён.")
            break

        if mode == "claude":
            print(generate_claude_reply(client, messages, user_text))
        else:
            answer = generate_openai_reply(client, messages, user_text)
            print(f"Ассистент: {answer}")
        print()


if __name__ == "__main__":
    mode = choose_mode()
    if mode == "claude":
        print("Включена думающая модель Claude.\n")
        client = make_claude_client()
        messages = new_claude_dialog()
    else:
        print("Включена обычная модель OpenAI.\n")
        client = make_openai_client()
        messages = new_openai_dialog()
    chat_loop(mode, client, messages)
