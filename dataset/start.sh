#!/bin/bash
chmod a+x ./download.sh
./download.sh 1_V7cTItws2gdBy7WKBUzaeqn0K339IUW ctr_dataset
unzip -j ctr_dataset 
rm -rf ctr_dataset
