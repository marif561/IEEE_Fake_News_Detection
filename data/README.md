# Data

The three datasets are public but not redistributed here. Download them and follow their licences.

| ID | Where | Layout expected by `hqnn.train` |
|---|---|---|
| **KG1** | Kaggle, "Fake and real news" (`Fake.csv`, `True.csv`; 23,481 fake / 21,417 real) | `--fake Fake.csv --real True.csv` |
| **KG2** | Kaggle, 6,335 articles (3,164 fake / 3,171 real) with a `label` column (`FAKE` / `REAL`) | `--data file.csv --label-col label` |
| **HF** | Hugging Face [ErfanMoosaviMonazzah/fake-news-detection-dataset-English](https://huggingface.co/datasets/ErfanMoosaviMonazzah/fake-news-detection-dataset-English) (15,478 fake / 14,522 real) | `--data file.csv --label-col label` |

Text columns are detected automatically (`title`, `text`, `content`, …) and joined; use `--text-cols` to choose them. Labels `fake`/`real`, `FAKE`/`REAL` and `1`/`0` (1 = fake) are understood. If your file uses other values, pass `--fake-value`. Check the dataset card for which number means fake.

Example layout for `scripts/run_all.sh`:

```
data/
├── kg1/  Fake.csv  True.csv
├── kg2/  fake_or_real_news.csv
└── hf/   train.csv
```

## Sample

[`sample/`](sample/) has 60 **invented** headlines (30 fake-style, 30 real-style) in both layouts, so the pipeline and tests run offline. They are not from any of the datasets.
