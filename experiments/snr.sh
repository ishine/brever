python scripts/compare_models.py -i \
models/3c7ffd87 \
models/c20e3c6b \
models/52674299 \
models/e9b58be9 \
models/b7033513 \
-t \
data/processed/test/7fdad118 \
data/processed/test/7bdd57ba \
data/processed/test/5ebcb5f3 \
data/processed/test/cfcbc381 \
--legend \
"-5 dB SNR" \
"0 dB SNR" \
"5 dB SNR" \
"10 dB SNR" \
"-5 -- 10 dB SNR" \
--xticks \
"-5 dB SNR" \
"0 dB SNR" \
"5 dB SNR" \
"10 dB SNR" \
--train-curve \
--output-dir \
pics/exp/snr \
