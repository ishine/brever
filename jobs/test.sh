#!/bin/sh

usage() { echo "Usage: $0 [inputs] [-f] [--njobs <int>]" 1>&2; exit 1; }

FORCE=''
NJOBS=1

while getopts ':f-:' optchar
do
    case ${optchar} in
        -)
            case "${OPTARG}" in
                njobs)
                    echo 123
                    NJOBS="${!OPTIND}"; OPTIND=$(($OPTIND+1)) ;;
                *) usage ;;
            esac;;
        f) FORCE='-f' ;;
        *) usage ;;
    esac
done

shift "$((OPTIND-1))"

echo "FORCE='${FORCE}'"
echo "NJOBS='${NJOBS}'"
echo "$@"

#for dir in "$@"
#do
#    echo "$dir"
#done
