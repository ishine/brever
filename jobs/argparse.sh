#!/bin/sh

OPTS=`getopt -o fn: -l force,njobs: -- "$@"`
if [ $? -ne 0 ]; then exit 1; fi
eval set -- "$OPTS"

FORCE=false
NJOBS=1

while true; do
  case "$1" in
    -f | --force ) FORCE=true; shift ;;
    -n | --njobs ) NJOBS="$2"; shift; shift ;;
    -- ) shift; break ;;
  esac
done
