#!/bin/sh
sed -e "s|COMMAND|$2|g" < $1 | bsub
