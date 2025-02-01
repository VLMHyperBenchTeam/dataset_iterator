# dataset_iterator

Пакет, реализующий итерирование по датасетам и прогон модели по нему.

Логически разделен на 2 части:

- Итераторы. Унаследованы от `AbstractIterator`. Реализуют итерирование по каждому из типов датасетов. 

- Раннеры. Унаследованы от `AbstractDatasetRunner`. Реализуют вызов модели на каждом элементе датасета. Элементы датасета получаем с помощью итераторов.


Для запуска `example.py` нужны данные из датасетов VQA и RPO. Моя структура каталога datasets выглядит следующим образом.
```
datasets
│
│   .gitignore
│
├───vqa
│   │   annotations.csv
│   │
│   └───images
│           0.jpg
│           1.jpg
│           2.jpg
│           3.jpg
│           4.jpg
│           5.jpg
│           6.jpg
│           7.jpg
│           8.jpg
│           9.jpg
│
└───rpo
    │   classes.json
    │
    ├───images
    │   ├───0
    │   │       0.jpg
    │   │       1.jpg
    │   │       2.jpg
    │   │       3.jpg
    │   │       4.jpg
    │   │       5.jpg
    │   │
    │   ├───1
    │   │       0.jpg
    │   │       1.jpg
    │   │       2.jpg
    │   │       3.jpg
    │   │       4.jpg
    │   │       5.jpg
    │   │
    │   ├───2
    │   │       0.jpg
    │   │       1.jpg
   ...
    │   │
    │   └───9
    │           0.jpg
    │           1.jpg
    │           2.jpg
    │
    └───jsons
            0.json
            1.json
            2.json
            3.json
            4.json
            5.json
            6.json
            7.json
            8.json
            9.json
```