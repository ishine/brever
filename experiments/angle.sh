python scripts/compare_models.py -i \
models/c20e3c6b \
models/a0a516cd \
-t \
data/processed/test/7bdd57ba \
data/processed/test/1b9937e1 \
--legend \
"fixed speaker location" \
"random speaker location" \
--xticks \
"fixed speaker location" \
"random speaker location" \
--train-curve \
--output-dir \
pics/exp/angle \
