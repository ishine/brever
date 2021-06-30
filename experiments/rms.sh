python scripts/compare_models.py -i \
models/663471e9 \
models/08f68fc9 \
-t \
data/processed/test/0fef80d2 \
data/processed/test/9392131b \
--legend \
"fixed speaker level" \
"random mixture level" \
--xticks \
"fixed speaker level" \
"random mixture level" \
--train-curve \
--output-dir \
pics/exp/rms \
