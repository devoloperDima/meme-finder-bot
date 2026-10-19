import sys
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry_ollama(fn: Callable[..., T], *args, attempts: int = 5, base_delay: float = 15.0) -> T:
    """Повтор запроса к Ollama при 5xx, таймауте или обрыве соединения: сервер отвечает 503,
    пока загружает модель, а запрос, попавший на его перезапуск, может не вернуться никогда.

    Другие ошибки (битый JSON, 4xx) пробрасываются сразу — повтор их не исправит.
    """
    import httpx
    import ollama

    for attempt in range(attempts):
        try:
            return fn(*args)
        except (ollama.ResponseError, httpx.TransportError) as e:
            server_side = isinstance(e, httpx.TransportError) or e.status_code >= 500
            if not server_side or attempt == attempts - 1:
                raise
            time.sleep(base_delay * (attempt + 1))
    raise AssertionError("unreachable")


def too_many_errors(errors: int, total: int) -> bool:
    """Массовые ошибки (>2%, но не меньше 10) — сигнал конвейеру, что шаг надо повторить."""
    return errors > max(10, 0.02 * total)


def keep_awake() -> None:
    """Просим Windows не уходить в сон по бездействию, пока жив этот процесс.

    Настройки питания не меняются: запрос снимается сам, когда процесс завершается.
    Нужно для ночных прогонов — иначе машина засыпает через 15 минут простоя.
    """
    if sys.platform == "win32":
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
