#!/bin/sh
sed -e "s|command|$2|g" < $1 | bsub
