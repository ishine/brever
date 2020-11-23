#!/bin/sh
source jobs/parse_args.sh

function parse_yaml {
    local prefix=$2
    local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
    sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
    awk -F$fs '{
        indent = length($1)/2;
        vname[indent] = $2;
        for (i in vname) {if (i > indent) {delete vname[i]}}
        if (length($3) > 0) {
            vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
            printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
        }
    }'
}

eval $(parse_yaml defaults.yaml)

for input in "$@"
do
    if [ -f $input/pesq_scores.mat ] && [ "$FORCE" == "" ]
    then
        echo "model already tested: $input "
    else
        bash jobs/send.sh jobs/job.sh 'python scripts/test_model.py '"$input$FORCE"'; matlab -nodisplay -nodesktop -nosplash -r "addpath matlab; addpath matlab/loizou; testModel '"$input $PRE_FS $PRE_MIXTURES_PADDING"'"; find '"$input"' -name "*.wav" -type f -delete'
    fi
done
