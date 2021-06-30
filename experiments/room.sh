python scripts/compare_models.py -i \
models/663471e9 \
models/d64d5d02 \
models/75a2c3af \
models/168af093 \
models/bad0c751 \
-t \
data/processed/test/0fef80d2 \
data/processed/test/6988091d \
data/processed/test/6ad55a49 \
data/processed/test/01f95c65 \
--legend \
"Room A" \
"Room B" \
"Room C" \
"Room D" \
"All rooms" \
--xticks \
"Room A" \
"Room B" \
"Room C" \
"Room D" \
--train-curve \
--output-dir \
pics/exp/room \
