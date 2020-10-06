#!/bin/sh
echo submitting "$1" containing the following command: "$2"
sed -e "s|command|$2|g" < $1 | bsub
