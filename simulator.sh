#!/bin/sh
t_begin=$(date +%s%3N)

./simulator.exe $1

t_end=$(date +%s%3N)
echo "print(($t_end - $t_begin) / 1000)" | python3 > cuspike-elapsedtime.txt
