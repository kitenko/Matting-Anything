#!/bin/bash

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <command>"
  exit 1
fi

COMMAND="$@"

docker-compose run --rm matting_anything $COMMAND
