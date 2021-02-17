#!/bin/sh
source jobs/parse_args.sh

for input in "$@"
do
    if [ -f $input/pesq_scores.mat ] && [ "$FORCE" == "" ]
    then
        echo "model already tested: $input "
    else
        bash jobs/send.sh jobs/job.sh "python scripts/test_model.py $input$FORCE"
    fi
done
