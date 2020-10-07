#!/bin/sh
source jobs/parse_args.sh

for input in "$@"
do
    bash jobs/send.sh jobs/job.sh "python scripts/train_model.py $input$FORCE"
done
