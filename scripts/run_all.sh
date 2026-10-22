#!/usr/bin/env bash
# Reproduce Table I: 5-fold cross-validation of the HQNN on KG1, KG2 and HF.
# Usage: bash scripts/run_all.sh <data_dir>
#   <data_dir>/kg1/Fake.csv + True.csv     (two-file layout)
#   <data_dir>/kg2/fake_or_real_news.csv   (label column FAKE / REAL)
#   <data_dir>/hf/train.csv                (label column; 1 = fake unless --fake-value is given)
set -euo pipefail
D=${1:?"data directory required"}
python -m hqnn.train --fake "$D/kg1/Fake.csv" --real "$D/kg1/True.csv" --name KG1 --baselines
python -m hqnn.train --data "$D/kg2/fake_or_real_news.csv" --label-col label --name KG2 --baselines
python -m hqnn.train --data "$D/hf/train.csv" --label-col label --name HF --baselines
