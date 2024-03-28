<div alight='left'>
  
# Sentiment Analysis Bot
Этот репозиторий содержит исходный код бота, который использует машинное обучение для определения тональности фраз. Бот способен классифицировать фразы на положительные, отрицательные и нейтральные. Он написан на Python с использованием фреймворка Rasa.
## Установка
Откройте терминал
  
__Windows__

`Win + R и введите cmd`

__Linux__

`Ctrl + Alt + T`

__Mac Os__

`Cmd + Пробел и введите iTerm`
### Совместимость 

Убедитесь что у вас установлены верные версии с помощью команд:

`python --version` # Python 3.8

`pip --version` # pip 19.2.3

### Клонирование репозитория

Склонируйте репозиторий с Github

`git clone https://github.com/alyaa17/rasa_sentiment.git`

### Установка зависимостей

Перейдите в директорию проекта и установите необходимые зависимости из файла requirements.txt:

`cd rasa_sentiment`

`pip install -r requirements.txt`

### Обучение модели

Бот использует машинное обучение для классификации фраз. Убедитесь, что модель обучена перед использованием:

`rasa train`
## Взаимодействие с ботом
Запустите сервер действий, чтобы бот мог их обрабатывать:

`rasa run actions`
В отдельном терминале можете начать чат с ботом:

`rasa shell`
</div>
