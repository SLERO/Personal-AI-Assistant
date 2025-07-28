import gradio as gr
from llama_cpp import Llama
import time

print("🚀 Инициализация Saiga Assistant...")

# Загружаем модель один раз при запуске
llm = Llama(
    model_path="/home/dragon/ai_assistant/saiga-7b-Q4_K_M/saiga-7b-Q4_K_M.gguf",
    n_ctx=1024,        # Уменьшаем контекст с 2048
    n_threads=6,       # Увеличиваем потоки CPU
    n_batch=512,       # Больше батч для эффективности
    verbose=False
)

print("✅ Модель загружена и готова к работе!")

def chat_with_saiga(message, history):
    """Функция для обработки сообщений пользователя с системным промптом"""
    print(f"👤 Пользователь: {message}")
    
    # Системный промпт для улучшения качества ответов
    system_prompt = "Ты - опытный программист и IT-специалист. Отвечай только на русском языке. Давай четкие, понятные и структурированные объяснения. Если вопрос касается программирования - приводи примеры кода."
    
    # Формируем полное сообщение с системным промптом
    full_message = f"{system_prompt}\n\nВопрос пользователя: {message}\n\nОтвет:"
    
    start_time = time.time()
    response = llm(
    full_message,
    max_tokens=150,    # Сокращаем длину ответов
    temperature=0.3,
    stop=["\n\n", "Вопрос:", "User:", "Пользователь:"]
)
    
    response_time = time.time() - start_time
    answer = response['choices'][0]['text'].strip()
    
    print(f"🤖 Saiga ({response_time:.1f}с): {answer}")
    return answer

# Минимальная конфигурация ChatInterface
interface = gr.ChatInterface(
    chat_with_saiga,
    title="🤖 Saiga Assistant",
    description="Русскоязычный ИИ-ассистент на базе модели Saiga 7B"
)

print("🌐 Запуск веб-интерфейса...")
print("📱 Откройте в браузере: http://localhost:7860")

interface.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False
)

EOF


