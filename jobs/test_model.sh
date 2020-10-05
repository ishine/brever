#!/bin/sh
source jobs/argparse.sh
for input in "$@"
do
    bash jobs/sender.sh jobs/job.sh "python scripts/test_model.py $input"
    matlab -nodisplay -r "addpath matlab; testModel $input"
done
