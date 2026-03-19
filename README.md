# DL Lab 2: Staff Detection

<p align="center">
  <img src="https://img.shields.io/badge/NumPy-Vectorized_Math-013243?style=flat&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/OpenCV-Image_Processing-5C3EE8?style=flat&logo=opencv&logoColor=white" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/YOLO-v11_|_v12_|_v26-5e8bde?style=flat&logo=ultralytics&logoColor=white" alt="YOLO"/>
  <img src="https://img.shields.io/badge/ReID-Person_Tracking-9932CC?style=flat" alt="ReID"/>
  <img src="https://img.shields.io/badge/OCR-Custom-FF1493?style=flat" alt="OCR"/>
</p>

### Структура:
```
.
├── assets/                     # Файлы для OCR и маппинга
│   ├── font.bin                # Бинарная маска шрифта камер для OCR
│   ├── loc2cam.json            # Маппинг локаций в камеры
│   └── uniloc.json             # Уникальные идентификаторы локаций
├── metadata/                   # Извлеченные и обработанные метаданные
│   ├── trn_meta.csv            # Метаданные размеченной выборки (таймстемпы, локации)
│   └── tst_meta.csv            # Метаданные тестовой выборки
├── preprocessing/              # Предобработка данных
│   ├── attn-mask/              # Генерация масок внимания
│   │   ├── attn-mask.ipynb     # Пайплайн вычисления масок
│   │   └── make-median.ipynb   # Расчет фоновых медиан для локаций
│   └── meta-extract/       
│       └── meta-extract.ipynb  # Быстрый парсинг таймстемпов с кадров
├── research/                   # Ноутбуки с экспериментами и R&D
│   ├── research-2*.ipynb       # Итерации разработки алгоритмов OCR
│   ├── research-[4,5].ipynb    # Итерации разработки алгоритмов масок внимания
├── scripts/                    # Вспомогательные скрипты
│   ├── utils/
│   │   └── restructurer.ipynb  # Утилиты для реструктуризации датасета
│   └── visualize/
│       ├── draw_rects.py       # Отрисовка боксов
│       ├── log-parser.ipynb    # Парсинг логов YOLO и построение графиков (Loss, mAP)
└── yolo/                       # Обучение и Инференс
    ├── train.ipynb             # Пайплайн обучения YOLO (train/val сплит, подготовка yaml)
    └── tracking.ipynb          # Инференс + REID Voting
```


