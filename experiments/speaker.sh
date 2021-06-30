python scripts/compare_models.py -i \
models/663471e9 \
models/288d425e \
-t \
data/processed/test/0fef80d2 \
data/processed/test/635843d2 \
--legend \
"ieee" \
"timit" \
--xticks \
"ieee" \
"timit" \
--train-curve \
--output-dir \
pics/exp/speaker \
