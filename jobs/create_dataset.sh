#!/bin/sh
source jobs/parse_args.sh

for input in "$@"
do
    if [ -f $input/dataset.hdf5 ] && [ "$FORCE" == "" ]
    then
        echo "dataset already created: $input "
    else
        bash jobs/send.sh jobs/job.sh "python scripts/create_dataset.py $input$FORCE"
    fi
done
