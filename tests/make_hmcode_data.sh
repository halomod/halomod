#!/usr/bin/env bash

# This file creates the hmcode_power.dat file used in tests

# Get the directory this script is in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

git clone https://github.com/alexander-mead/hmcode /tmp/hmcode
cd /tmp/hmcode
sed -i s/ihm=1/ihm=2/g HMcode.f90
./compile.sh
./HMcode.e
cp power.dat $DIR/data/hmcode_power.dat
