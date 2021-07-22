python scripts/compare_models.py -i \
models/c20e3c6b \
models/b8817980 \
-t \
data/processed/test/7bdd57ba \
data/processed/test/9d2d9436 \
--legend \
"fixed speaker level" \
"random mixture level" \
--xticks \
"fixed speaker level" \
"random mixture level" \
--train-curve \
--output-dir \
pics/exp/rms \
