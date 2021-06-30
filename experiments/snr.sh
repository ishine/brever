python scripts/compare_models.py -i \
models/825fc37d \
models/663471e9 \
models/af7b9f63 \
models/01d3d4f1 \
models/83911bb4 \
-t \
data/processed/test/4790e113 \
data/processed/test/0fef80d2 \
data/processed/test/3abc4898 \
data/processed/test/698678a5 \
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
