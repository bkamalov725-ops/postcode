# PostCode — оценка загрузки транспорта по фотографии

Команда **королевскийбумбет**: Камалов Булат, Азизмурадов Ринат.

MVP принимает фотографию грузового отсека и номер перевозки, оценивает заполненность
от 0 до 100%, сохраняет результат в SQLite и возвращает XML. Есть веб-интерфейс для демонстрации,
история оценок и скрипт повторного обучения на подтверждённой разметке.
Целевая величина — экспертная оценка заполнения по размещённым объектам, условно по занятой площади пола.
Масса и использование грузоподъёмности не вычисляются.

## Выбранная модель и результаты

**Attention518 / DINOv2 ViT-B/14, сохранённый повторный прогон baseline_recovery.**

| Результат | Значение | Что именно измерено |
|---|---:|---|
| Исторический Attention518 | 5.4545 | Public MAE первого прогона; его точные обученные веса потеряны |
| Сохранённые повторные варианты | около 5.46 | Участник сообщил 5.4591 / 5.4599 без однозначного соответствия двум CSV |
| Развёрнутый baseline_recovery | 5.9028 | Primary OOF MAE |
| Развёрнутый baseline_recovery | 5.9255 | Secondary OOF MAE |

MAE измеряется в процентных пунктах; меньше — лучше. Primary и secondary — два групповых
разбиения одних обучающих данных, не независимые тестовые выборки. Private-результат неизвестен.
Точное повторение исходных 5.4545 не заявляется. Ветка Continue4 с результатом 5.4483
не используется в выбранном сервисе по решению команды.

Эволюция public-результатов: DINOv2 + SVR **9.4549** → дообучение DINOv2-B **≈5.93** →
порядковая модель **≈5.69** → Attention392 **≈5.6** → Attention518 **5.4545**.
Округлённые числа взяты из сообщений команды; OOF сохранённого повтора — из `report.json` в архиве модели.

## Запуск с отдельно полученными весами

Нужны Git и Python **3.12**. Эта версия GitHub содержит исходный код, документацию и презентацию; датасеты train/validation/test и обученные веса не включены. Для инференса предварительно получите архив `model_complete_primary.zip` у команды. Без него сервис с обученной моделью не запустится. Команды выполняются из корня репозитория.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe scripts/install_attention518.py "C:/path/to/model_complete_primary.zip"
.\.venv\Scripts\python.exe serve.py
```

Открыть **http://127.0.0.1:8000**, выбрать фото и указать номер перевозки.
Демонстрационное фото: [tests/fixtures/cargo_sample.jpg](tests/fixtures/cargo_sample.jpg).
На проверенном CPU оно даёт примерно **38.03%**; это прогноз, а не известная тестовая разметка.
Если среда и `runtime/attention518` уже подготовлены, достаточно `python serve.py` в этой среде.
Установщик намеренно не перезаписывает существующий каталог модели.

Linux/macOS: вместо `.venv\Scripts\python.exe` используйте `.venv/bin/python`.
При установленной совместимой CUDA-сборке PyTorch можно запустить `python serve.py --device cuda:0`.
Интернет нужен для первоначальной установки зависимостей; после установки весов инференс работает офлайн.

## Веса предоставляются отдельно

Архив модели не включён в эту облегчённую публикацию. Требуется `model_complete_primary.zip`, 659303075 байт.
SHA-256: `68894a977620a3c5b0ac07f0677e79ebcc3f9941d3b517d7c31f49c88071f43a`.
Архив содержит базовый энкодер, пять обученных моделей, исходник DINOv2 с лицензией, код и конфигурации.
Установщик проверяет контрольную сумму. Подробнее: [models/README.md](models/README.md).

## Как устроен прогноз

Фото → EXIF-ориентация и RGB → сохранение пропорций и padding до **518×518** → нормализация ImageNet →
DINOv2 ViT-B/14 → обучаемое внимание по пяти пространственным областям → порядковая голова.
Дообучаются два последних блока DINOv2. Голова выдаёт 20 монотонных вероятностей уровней загрузки.
Ожидаемая загрузка равна `5 × sum(probabilities)`. Усредняются оригинал и горизонтальное отражение,
затем пять primary-fold моделей. **Нет дополнительного округления или калибровки.**
Attention-веса не являются подтверждённой сегментацией груза или пола.

## XML API

`POST /assessments`, `Content-Type: application/xml`:

```xml
<assessment>
  <shipment_id>DEMO-001</shipment_id>
  <image encoding="base64">BASE64_JPEG_OR_PNG</image>
</assessment>
```

Ответ **201**, XML `<assessment>`: `assessment_id`, `shipment_id`, `load_pct`, `model_version`,
`created_at`, `processing_ms`. [Реальный пример ответа](docs/final_service_example.xml).

| Метод | Назначение |
|---|---|
| `POST /assessments` | Обработать фото и сохранить новую оценку |
| `GET /shipments/{shipment_id}` | Получить последнюю оценку |
| `GET /shipments/{shipment_id}/history?limit=100` | Получить историю оценок |
| `GET /health` | Проверить готовность сервиса |

Повторный POST добавляет запись в историю, не перезаписывая предыдущую. Результаты переживают перезапуск:
база находится в `runtime/assessments.sqlite3`. Фото удаляется из временного каталога после обработки;
сохраняются результат, номер, версия модели, время и SHA-256 снимка.

Поддерживаются JPEG/PNG до 10 МиБ и до 20 мегапикселей; XML до 16 МиБ.
Ошибки тоже XML: **400** — некорректные данные; **404** — перевозка не найдена;
**413** — превышен размер; **415** — неподдерживаемый формат.
DTD и XML entities запрещены. Полный контракт: [docs/API.md](docs/API.md).

## Воспроизведение CSV

[submission/candidate_attention518_restored.csv](submission/candidate_attention518_restored.csv)
содержит 307 прогнозов сохранённого повторного прогона. Для пересчёта по изображениям:

```powershell
python scripts/predict_attention518.py --images-dir data/test/images --ids-csv data/test/test.csv --output outputs/reproduced_attention518.csv --device cpu
```

`test.csv`: колонка `image_id`; имена изображений `<image_id>.jpg`. Данные предоставляет организатор,
в репозиторий они не включены. Прогнозы рассчитываются заново, не подставляются из готового CSV.
CPU и CUDA mixed precision могут давать небольшие численные различия.
Полный локальный повтор этой версии на всех 307 фото ещё не выполнен.

## Повторное обучение на новых данных

Подготовьте `train.csv` с колонками `image_id,load_pct`, `train_groups.csv` с колонками
`image_id,group_id` и JPG-файлы. Метки должны быть подтверждены экспертом.
Снимки одной группы остаются целиком в одной части разбиения.

```powershell
python scripts/retrain_attention518.py --train-csv data/train.csv --groups-csv data/train_groups.csv --images-dir data/images --output outputs/retrained --plan-only
python scripts/retrain_attention518.py --train-csv data/train.csv --groups-csv data/train_groups.csv --images-dir data/images --output outputs/retrained --device cuda:0
```

Первый вызов проверяет данные и формирует план без обучения. Второй использует исходную функцию обучения:
5 групповых folds, pretrained-инициализация отдельно в каждой части, 12 эпох,
среднее последних трёх снимков параметров, `MAE(load/100) + 0.2 × ordinal BCE`.
Сохраняются промежуточные состояния для возобновления, OOF, метрики и новый переносимый bundle.
Прежние веса не меняются. После проверки новой версии:
`python serve.py --bundle-dir outputs/retrained`.
Новая версия не наследует старую leaderboard-оценку.

## Проверки

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests/test_service.py -q
python scripts/check_final_service.py
```

Полный набор: **85 тестов и 102 дополнительных сценария прошли** (включая 19 API-тестов).
Отдельно проверена реальная фотография → модель → XML → SQLite →
получение после перезапуска. [Отчёт](docs/final_service_smoke.json).
CPU-инференс на одном проверенном снимке — около **19–20 секунд**; это не нагрузочный тест и не SLA.
Подготовка плана повторного обучения проверена на 716 фото. Новый полный GPU-прогон не запускался.

## Docker

Сначала установите архив модели в `runtime/attention518`.

```powershell
docker build -t postcode .
docker run --rm -p 8000:8000 -v "${PWD}/runtime:/app/runtime" postcode
```

Dockerfile подготовлен, но сборка контейнера здесь не проверялась. Runtime-том хранит и веса, и базу.
Локальный сервер по умолчанию слушает 127.0.0.1; контейнер — 0.0.0.0.

## Презентация и состав проекта

- [PPTX, 8 слайдов](docs/presentation/PostCode_final_presentation.pptx)
- [PDF для сдачи, менее 20 МБ](docs/presentation/PostCode_final_presentation.pdf)
- `postcode_ml/service.py`, `storage.py` — XML API и история.
- `postcode_ml/best_model.py` — адаптер сохранённого Attention518.
- `postcode_ml/web/index.html` — интерфейс демонстрации.
- `scripts/install_attention518.py`, `predict_attention518.py`, `retrain_attention518.py` — веса, прогноз и обучение.
- `models/` — инструкция получения и установки отдельно предоставляемых весов.

Прежние `run_experiment.py`, `run_improved.py`, `predict_bundle.py`, notebooks и SVR-модули
оставлены для истории экспериментов; для финального решения используйте команды этого README.
[Описание первоначального SVR-решения](docs/history/INITIAL_SVR_README.md).

MVP не включает производственную авторизацию, промышленную интеграцию, видеопоток или автоматический выбор машины.
GitLab CI/CD и runners не используются согласно правилам хакатона. Отправка кода — через рабочую ветку
и Merge Request в защищённую `main`.
