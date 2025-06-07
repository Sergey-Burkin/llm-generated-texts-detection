# llm-generated-texts-classification

## Описание задачи

Классификация текстов. На вход подается строка, на выходе - вероятность того,
что текст был сгенерирован нейросетью или написан человеком.

Для тренировки был использован
[датасет](https://www.kaggle.com/datasets/conjuring92/ai-mix-v26/data), который
является дополнением оригинального из
[соревнования](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/)

## Технические детали

В этом проекте были использованы следующие пакеты:

- poetry - для управления зависимостями
- pre-commit - для контроля качества кода
- PyTorch Lightning - для тренировки модели
- dvc и google drive - для хранения данных
- hydra - для управления конфигами
- mlflow - для логгирования метрик
- fire - для создания CLI

## Запуск проекта

### Подготовка окружения

1.  `git clone https://github.com/Sergey-Burkin/llm-generated-texts-detection.git`
2.  `cd llm-generated-texts-detection`
3.  Скретный ключ от gdrive положить в папку `.dvc`
4.  `docker compose build [classifier-model]` - по умолчанию docker compose еще
    собирает и запускает сервер mlflow по адресу `127.0.0.1:8080`
5.  `docker compose up [classifier-model] -d`
6.  `docker exec -it classifier_model bash`
7.  `conda activate dev`
8.  `dvc pull`

### Тренировка модели

`python3 commands.py train`

Информация о параметрах обучения в `configs/config.yaml`. Графики отправляются в
mlflow

### Запуск модели

`python3 commands.py infer --text="Sample text."`
