"""День 5: Telegram-бот. Пишешь ситуацию — бот присылает подходящий мем.

Под мемом кнопки 👍 / 👎 / 🔄 «другой» — все показы и реакции пишутся в SQLite
(memebot.bot.storage) для онлайн-метрик и следующих итераций обучения.

  python -m memebot.bot.main
"""

import asyncio
import hashlib
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import CallbackQuery, FSInputFile, InlineKeyboardButton, InlineKeyboardMarkup, Message

from memebot.bot.storage import Storage
from memebot.config import settings
from memebot.search.service import MemeSearcher
from memebot.utils import keep_awake

MAX_RESULTS = 10
ACTIONS = {"like", "dislike", "next"}
GREETING = (
    "Привет! Опиши ситуацию — пришлю подходящий мем.\n\n"
    "Например: «когда препод говорит, что контрольная будет лёгкой» или «понедельник, снова на работу».\n\n"
    "Под мемом кнопки: 👍 зашёл, 👎 мимо, 🔄 другой. Оценки помогают боту стать смешнее."
)

dp = Dispatcher()
searcher: MemeSearcher
storage: Storage


def keyboard(impression_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="👍", callback_data=f"fb:{impression_id}:like"),
        InlineKeyboardButton(text="👎", callback_data=f"fb:{impression_id}:dislike"),
        InlineKeyboardButton(text="🔄 другой", callback_data=f"fb:{impression_id}:next"),
    ]])


async def send_meme(message: Message, user_id: int, search_id: int, rank: int) -> None:
    results = storage.results(search_id)
    if rank >= len(results):
        await message.answer("Мемы по этому запросу закончились 🙃 Попробуй описать ситуацию иначе.")
        return
    meme_id = results[rank]
    impression_id = storage.log_impression(search_id, user_id, meme_id, rank)
    photo = storage.cached_file_id(meme_id) or FSInputFile(settings.data_dir / searcher.image_path[meme_id])
    sent = await message.answer_photo(photo, reply_markup=keyboard(impression_id))
    if isinstance(photo, FSInputFile):  # повторно не загружаем: Telegram отдаёт file_id
        storage.cache_file_id(meme_id, sent.photo[-1].file_id)


@dp.message(CommandStart())
async def start(message: Message) -> None:
    await message.answer(GREETING)


@dp.message(F.text)
async def on_text(message: Message) -> None:
    query = message.text.strip()[:300]
    hits, timings = await asyncio.to_thread(searcher.search, query, MAX_RESULTS)
    search_id = storage.log_search(message.from_user.id, query, [h.meme_id for h in hits], timings)
    logging.info("search %d %.0f ms", search_id, sum(timings.values()))
    await send_meme(message, message.from_user.id, search_id, 0)


@dp.callback_query(F.data.startswith("fb:"))
async def on_feedback(callback: CallbackQuery) -> None:
    parts = callback.data.split(":")
    if len(parts) != 3 or not parts[1].isdigit() or parts[2] not in ACTIONS:
        await callback.answer()
        return
    impression_id, action = int(parts[1]), parts[2]
    impression = storage.impression(impression_id, callback.from_user.id)
    if impression is None or not storage.log_feedback(impression_id, action):
        await callback.answer()
        return
    search_id, _, rank = impression
    await callback.message.edit_reply_markup(reply_markup=None)  # одна реакция на показ
    if action == "next":
        await callback.answer()
        await send_meme(callback.message, callback.from_user.id, search_id, rank + 1)
    else:
        await callback.answer("Спасибо! 🙌" if action == "like" else "Учту 👌")


async def main() -> None:
    global searcher, storage
    keep_awake()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not settings.telegram_bot_token:
        raise SystemExit("TELEGRAM_BOT_TOKEN не задан: впишите токен от @BotFather в .env")
    salt = settings.bot_user_salt or hashlib.sha256(settings.telegram_bot_token.encode()).hexdigest()
    storage = Storage(settings.bot_db_path, salt)
    logging.info("Загружаю поиск (%s)...", settings.search_model)
    searcher = MemeSearcher(settings.search_model, device=settings.search_device)
    bot = Bot(settings.telegram_bot_token)
    logging.info("Бот запущен")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
