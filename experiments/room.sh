python scripts/compare_models.py -i \
models/c20e3c6b \
models/9b695a2b \
models/bca1563e \
models/f38cf5e6 \
models/aa36852f \
models/682c2400 \
models/10866055 \
models/b32efeea \
models/53abdb09 \
models/ca22f391 \
-t \
data/processed/test/7bdd57ba \
data/processed/test/bdc36672 \
data/processed/test/f3298ee1 \
data/processed/test/f09145fb \
data/processed/test/5e75d93a \
data/processed/test/7ae4d206 \
data/processed/test/aa6c2114 \
data/processed/test/2043d6df \
data/processed/test/b934a7e0 \
data/processed/test/0e4b85f8 \
--legend \
"SURREY A" \
"SURREY B" \
"SURREY C" \
"SURREY D" \
"SURREY all" \
"ASH 01" \
"ASH 01-09" \
"ASH all but 01-09" \
"ASH all but 01" \
"ASH all" \
--xticks \
"SURREY A" \
"SURREY B" \
"SURREY C" \
"SURREY D" \
"SURREY all" \
"ASH 01" \
"ASH 01-09" \
"ASH all but 01-09" \
"ASH all but 01" \
"ASH all" \
--train-curve \
--output-dir \
pics/exp/room \
