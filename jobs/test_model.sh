#!/bin/sh
source jobs/parse_args.sh

for input in "$@"
do
    # if [ -f $input/scores.json ] && [ "$FORCE" == "" ]
    if false
    then
        echo "model already tested: $input "
    else
        bash jobs/send.sh jobs/job_hpc.sh "python scripts/test_model.py $input$FORCE --no-cuda"
    fi
done
