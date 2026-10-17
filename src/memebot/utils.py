import sys


def keep_awake() -> None:
    """Просим Windows не уходить в сон по бездействию, пока жив этот процесс.

    Настройки питания не меняются: запрос снимается сам, когда процесс завершается.
    Нужно для ночных прогонов — иначе машина засыпает через 15 минут простоя.
    """
    if sys.platform == "win32":
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
