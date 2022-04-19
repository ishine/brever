#!/bin/sh

OPTS=$(getopt \
    --options fvq:c:m:w:g \
    --longoptions force,verbose,queue:,cores:,memory:,walltime:,gpu32gb \
    -- "$@"
)
if [ $? -ne 0 ]; then exit 1; fi
eval set -- "$OPTS"

FORCE=false
VERBOSE=false
QUEUE="hpc"
CORES=1
MEMORY=8
WALLTIME=24
GPU32GB=false

while true; do
  case "$1" in
    -f | --force ) FORCE=true; shift ;;
    -v | --verbose ) VERBOSE=true; shift ;;
    -q | --queue ) QUEUE="$2"; shift; shift ;;
    -c | --cores ) CORES="$2"; shift; shift ;;
    -m | --memory ) MEMORY="$2"; shift; shift ;;
    -w | --walltime ) WALLTIME="$2"; shift; shift ;;
    -g | --gpu32gb ) GPU32GB=true; shift ;;
    -- ) shift; break ;;
  esac
done

if [ ${QUEUE} != "hpc" ] && [ ${QUEUE} != "gpuv100" ] && [ ${QUEUE} != "gpua100" ]
then
    echo "requested wrong queue: ${QUEUE}"
    exit 1
fi

JOBFILE="jobs/job.sh"
echo "#!/bin/sh" > ${JOBFILE}
echo "#BSUB -q ${QUEUE}" >> ${JOBFILE}
echo "#BSUB -J jobname" >> ${JOBFILE}
echo "#BSUB -n ${CORES}" >> ${JOBFILE}
echo "#BSUB -W ${WALLTIME}:00" >> ${JOBFILE}
echo "#BSUB -R \"rusage[mem=${MEMORY}GB]\"" >> ${JOBFILE}
echo "#BSUB -R \"span[hosts=1]\"" >> ${JOBFILE}
echo "#BSUB -oo jobs/logs/%J.out" >> ${JOBFILE}
echo "#BSUB -eo jobs/logs/%J.err" >> ${JOBFILE}
if [ ${QUEUE} == "gpuv100" ] || [ ${QUEUE} == "gpua100" ]
then
    echo "#BSUB -gpu \"num=1:mode=exclusive_process\"" >> ${JOBFILE}
    if [ ${GPU32GB} = true ]
    then
        echo "#BSUB -R \"select[gpu32gb]\"" >> ${JOBFILE}
    fi
fi

echo "source venv/bin/activate" >> ${JOBFILE}
echo "COMMAND" >> ${JOBFILE}

if [ ${VERBOSE} = true ]
then
    echo "The following job template was created:"
    echo "---Beginning of file---"
    cat ${JOBFILE}
    echo "---End of file---"
fi

if [ $# -lt 2 ]
then
    echo "ERROR: at least 2 positional arguments are required"
    exit 1
fi

COMMAND="python scripts/test_model.py $1 ${@:2}"
if [ ${FORCE} = true ]
then
    COMMAND="${COMMAND} -f"
fi
if [ ${VERBOSE} = true ]
then
    echo "Submitting \"${COMMAND}\" as COMMAND"
fi
bash jobs/submit.sh jobs/job.sh "${COMMAND}"
