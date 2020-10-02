#!/bin/sh
sed -e "s|script|$2|g" -e "s|args|$3|g" < $1 | bsub
