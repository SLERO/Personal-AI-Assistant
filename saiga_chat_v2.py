#Добавлено сопряжение с ChromaDB для реализации RAG

import gradio as gr
from llama_cpp import Llama
import time
import chromadb
from chromadb.utils import embedding_functions
import datetime
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="gradio.chat_interface")

print("🚀 Инициализация Saiga Assistant...")

# Инициализация ChromaDB
client = chromadb.PersistentClient(path="./chroma_memory_db")
embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="intfloat/multilingual-e5-small")
collection = client.get_or_create_collection(
    name="saiga_memory",
    embedding_function=embed_func,
    metadata={"hnsw:space": "cosine"}
)

# Загружаем модель
llm = Llama(
    model_path="/home/dragon/ai_assistant/saiga-7b-Q4_K_M/saiga-7b-Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=6,
    n_batch=512,
    verbose=False
)

print("✅ Модель и БД готовы!")

def chat_with_saiga(message, history):
    print(f"👤 Пользователь: {message}")
    
    # Системный промпт с анти-мусором
    system_prompt = "Ты — прямолинейный чувак, как я: отвечай честно, без цензуры, без подлизывания. Манера: неформальная, с 'бро' и 'чувак'. Если дерьмо — скажи прямо. Не выводи даты, 'Дата:' или 'Прошлое:' в ответе — используй контекст только для мышления, суммируй по сути. Пример: если говорят 'Превед, медвед!', отвечай 'Превед, кросафчег!'."
    
    # Короткий history: последние 2, чтоб не зацикливаться
    short_history = "\n".join([h['content'] for h in history[-2:] if 'content' in h])
    
    # Поиск в БД: топ-2 (меньше, чтоб не перегружать)
    current_time = time.time()
    five_min_ago = current_time - 300
    results = collection.query(
        query_texts=[message],
        n_results=2,  # Уменьшили для чистоты
        where={"timestamp": {"$lt": five_min_ago}}
    )
    context = ""
    if results['documents'] and results['documents'][0]:
        # Суммируем контекст, чтоб не копировать дословно
        context = " ".join([doc.split('\n')[1].replace("Ассистент: ", "") for doc in results['documents'][0]])  # Только суть ответа
        print(f"DEBUG: Суммированный контекст: {context}")
    
    # Для "год назад" (суммированный)
    if "год назад" in message.lower():
        year_ago = current_time - 31536000
        results = collection.query(
            query_texts=[message],
            n_results=2,
            where={"timestamp": {"$gt": year_ago - 86400, "$lt": year_ago + 86400}}
        )
        context = " ".join([doc.split('\n')[1].replace("Ассистент: ", "") for doc in results['documents'][0]]) if results['documents'] else "Ничего не нашел год назад, чувак."
        print(f"DEBUG: Суммированный год назад: {context}")

    full_message = f"{system_prompt}\n\nКороткая история чата: {short_history}\n\nСуммированная прошлая история: {context}\n\nВопрос: {message}\n\nОтвет:"
    
    start_time = time.time()
    response = llm(
        full_message,
        max_tokens=200,  # Баланс: достаточно для полных, без безумия
        temperature=0.5,  # Для разнообразия
        stop=["\n\n"],  # Минимальный, чтоб не обрубало
        echo=False,  # Без эха промпта
        repeat_penalty=1.5  # Сильный штраф за повторы
    )
    response_time = time.time() - start_time
    answer = response['choices'][0]['text'].strip()
    
    # Сохранение (только суть, без лишнего)
    doc_id = f"msg_{int(current_time)}"
    current_date = datetime.datetime.fromtimestamp(current_time).strftime("%Y-%m-%d")
    collection.add(
        documents=[answer],  # Сохраняем только ответ, чтоб не засорять БД
        metadatas=[{"date": current_date, "timestamp": int(current_time), "user": "you", "original_message": message}],
        ids=[doc_id]
    )
    
    print(f"🤖 Saiga ({response_time:.1f}с): {answer}")
    return answer

# Gradio
interface = gr.ChatInterface(
    chat_with_saiga,
    chatbot=gr.Chatbot(type="messages", height=800),
    title="🤖 Saiga Assistant с ChromaDB",
    description="Русскоязычный ИИ с долговременной памятью"
)

print("🌐 Запуск веб-интерфейса...")
print("📱 Откройте в браузере: http://localhost:7861")

interface.launch(
    server_name="0.0.0.0",
    server_port=7861,
    share=False
)