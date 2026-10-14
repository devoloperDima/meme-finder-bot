"""Проверка окружения: GPU, CUDA в PyTorch, внешние инструменты и сервисы."""

import shutil
import subprocess
import sys
import urllib.request


def ok(name: str, detail: str = "") -> None:
    print(f"[ OK ] {name} {detail}")


def fail(name: str, detail: str = "") -> None:
    print(f"[FAIL] {name} {detail}")


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()


def main() -> None:
    v = sys.version_info
    (ok if (3, 11) <= v[:2] <= (3, 12) else fail)("Python", f"{v.major}.{v.minor}.{v.micro} (нужен 3.11–3.12)")

    for tool in ("nvidia-smi", "git", "ollama"):
        path = shutil.which(tool)
        if not path:
            fail(tool, "не найден")
            continue
        version = run([tool, "--version"]) if tool != "nvidia-smi" else run(
            [tool, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
        )
        ok(tool, version.splitlines()[0] if version else "")

    try:
        import torch

        if torch.cuda.is_available():
            ok("torch.cuda", f"{torch.__version__}, {torch.cuda.get_device_name(0)}")
        else:
            fail("torch.cuda", f"{torch.__version__}: CUDA недоступна (поставлена CPU-сборка?)")
    except ImportError:
        fail("torch", "не установлен")

    url = "http://localhost:11434/api/version"
    try:
        with urllib.request.urlopen(url, timeout=2):
            ok("Ollama API", url)
    except OSError:
        fail("Ollama API", f"{url} не отвечает")


if __name__ == "__main__":
    main()
