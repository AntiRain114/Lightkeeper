#!/bin/sh
# This script runs take_picture.py every 3 seconds indefinitely

while true
do
  /usr/bin/python3 take_picture.py
  sleep 3  # Sleep for 3 seconds before running the script again
done