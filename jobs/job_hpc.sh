#!/bin/sh
### General options
### -- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J jobname
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- set walltime limit: hh:mm
#BSUB -W 24:00
# request 4GB of system-memory per core
#BSUB -R "rusage[mem=4GB]"
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo jobs/logs/%J.out
#BSUB -eo jobs/logs/%J.err
# -- end of LSF options --

source venv/bin/activate
command
