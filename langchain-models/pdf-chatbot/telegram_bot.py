"""
Telegram frontend for the PDF chatbot RAG chain.

Reuses the FAISS index + chain built in app.py — no re-upload, no new DB.
Run: python telegram_bot.py
"""

import os
import logging

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

from app import chain

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Ask me anything about the PDF. Just type your question."
    )


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text.strip()
    if not question:
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        answer = chain.invoke(question)
    except Exception:
        logger.exception("Chain invoke failed")
        await update.message.reply_text("Something went wrong answering that. Try again.")
        return

    await update.message.reply_text(answer)


def main():
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set in .env")

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    logger.info("Bot starting (polling)...")
    app.run_polling()


if __name__ == "__main__":
    main()
