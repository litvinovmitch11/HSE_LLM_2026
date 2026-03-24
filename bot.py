import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Set

from telegram import Bot
from telegram.error import TelegramError
import aiohttp

from config import (
    BOT_TOKEN, SOURCE_CHANNEL_ID, 
    TARGET_CHAT_IDS, CHECK_INTERVAL
)
from model import model

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class TelegramMLBot:
    def __init__(self):
        self.bot = Bot(token=BOT_TOKEN)
        self.processed_messages: Set[int] = set()  # Храним ID обработанных сообщений
        self.last_check_id = None  # Последний обработанный message_id
        
    async def get_channel_messages(self, limit: int = 10):
        messages = []
        try:
            # Получаем обновления из канала
            updates = await self.bot.get_updates(
                allowed_updates=['channel_post'],
                offset=-1,
                limit=limit
            )
            
            for update in updates:
                if update.channel_post and update.channel_post.chat_id == SOURCE_CHANNEL_ID:
                    messages.append(update.channel_post)
                    
        except TelegramError as e:
            logger.error(f"Ошибка получения сообщений: {e}")
            
        return messages
    
    async def process_message(self, message):
        try:
            # Получаем текст сообщения
            text = message.text or message.caption or ""
            
            if not text:
                return None
            
            # Прогоняем через ML модель
            result = model.predict(text)
            
            return {
                'original_message_id': message.message_id,
                'original_text': text,
                'ml_result': result,
                'date': message.date
            }
            
        except Exception as e:
            logger.error(f"Ошибка обработки сообщения {message.message_id}: {e}")
            return None
    
    async def send_newsletter(self, results: List[Dict]):
        if not results:
            return
        
        # Формируем сообщение для рассылки
        message_text = "Новые результаты обработки:\n\n"
        for result in results:
            message_text += f"*Сообщение:*\n{result['original_text'][:1000]}...\n"
            message_text += f"*Результат ML:*\n{result['ml_result']}\n"
            message_text += f"*Время:* {result['date']}\n"
        
        # Отправляем каждому получателю
        for chat_id in TARGET_CHAT_IDS:
            try:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=message_text,
                    parse_mode='Markdown'
                )
                logger.info(f"Рассылка отправлена в чат {chat_id}")
            except TelegramError as e:
                logger.error(f"Ошибка отправки в чат {chat_id}: {e}")
    
    async def check_and_process(self):
        try:
            # Получаем новые сообщения
            messages = await self.get_channel_messages(limit=50)
            
            # Фильтруем только новые сообщения
            new_messages = [
                msg for msg in messages 
                if msg.message_id not in self.processed_messages
            ]
            
            if not new_messages:
                return
            
            logger.info(f"Найдено {len(new_messages)} новых сообщений")
            
            # Обрабатываем каждое сообщение
            results = []
            for message in new_messages:
                result = await self.process_message(message)
                if result:
                    results.append(result)
                self.processed_messages.add(message.message_id)
            
            # Если есть результаты - делаем рассылку
            if results:
                await self.send_newsletter(results)
                
        except Exception as e:
            logger.error(f"Ошибка в основном цикле: {e}")
    
    async def run(self):
        logger.info("Бот запущен и начал прослушивание...")
        
        while True:
            await self.check_and_process()
            await asyncio.sleep(CHECK_INTERVAL)

async def main():
    bot = TelegramMLBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
