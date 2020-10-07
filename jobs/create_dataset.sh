#!/bin/sh
source jobs/parse_args.sh

for input in "$@"
do
    bash jobs/send.sh jobs/job.sh "python scripts/create_dataset.py $input$FORCE"
done
