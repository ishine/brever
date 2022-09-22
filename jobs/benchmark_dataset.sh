#!/bin/sh

OPTS=$(getopt \
    --options vq:c:m:w:ga: \
    --longoptions verbose,queue:,cores:,memory:,walltime:,gpu32gb,args: \
    -- "$@"
)
if [ $? -ne 0 ]; then exit 1; fi
eval set -- "$OPTS"

VERBOSE=false
QUEUE="hpc"
CORES=1
MEMORY=8
WALLTIME=24
GPU32GB=false
ARGS=""

while true; do
  case "$1" in
    -v | --verbose ) VERBOSE=true; shift ;;
    -q | --queue ) QUEUE="$2"; shift; shift ;;
    -c | --cores ) CORES="$2"; shift; shift ;;
    -m | --memory ) MEMORY="$2"; shift; shift ;;
    -w | --walltime ) WALLTIME="$2"; shift; shift ;;
    -g | --gpu32gb ) GPU32GB=true; shift ;;
    -a | --args ) ARGS="$2"; shift; shift ;;
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
echo "#BSUB -oo jobs/logs/benchmark_%J.out" >> ${JOBFILE}
echo "#BSUB -eo jobs/logs/benchmark_%J.err" >> ${JOBFILE}
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


COMMAND="python scripts/benchmark_dataset.py ${1} ${ARGS}"
if [ ${VERBOSE} = true ]
then
    echo "Submitting \"${COMMAND}\" as COMMAND"
fi
bash jobs/submit.sh jobs/job.sh "${COMMAND}"
