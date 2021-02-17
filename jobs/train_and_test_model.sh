#!/bin/sh
source jobs/parse_args.sh

for input in "$@"
do
    if [ -f $input/config_full.yaml ] && [ "$FORCE" == "" ]
    then
        echo "model already trained: $input "
    else
        bash jobs/send.sh jobs/job.sh "python scripts/train_model.py $input$FORCE; python scripts/test_model.py $input$FORCE"
    fi
done
