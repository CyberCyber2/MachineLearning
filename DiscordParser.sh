#!/bin/bash

awk '
  $1 ~ /^\[/ && $3 ~ /\]$/ && $NF ~ "#" {
    sub(/\[.*\] /, "")
    sub("#.*", ":")
  }
  {print}
' combined.txt >> output.txt
