#!/bin/sh
source jobs/parse_args.sh

for input in "$@"
do
    if [ -f $input/mixture_info.json ] && [ "$FORCE" == "" ]
    then
        echo "dataset already created: $input "
    else
        bash jobs/send.sh jobs/job_hpc.sh "python scripts/create_dataset.py $input$FORCE"
    fi
done
