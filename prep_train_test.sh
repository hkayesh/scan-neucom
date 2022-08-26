#!/usr/bin/env bash
MODEL=scan

BASEDIR=./model_output/$MODEL
PROCESSDATADIR=$BASEDIR/data/
DATASETFILE=dataset_file_$MODEL.pkl

python ./prep.py -P $PROCESSDATADIR -o $DATASETFILE -v 0.1 -r asu_fullanno,chop_fullanno,smm4h_75_25 -t asu_fullanno,chop_fullanno,smm4h_75_25

WORKING_DIR=$BASEDIR/
mkdir -p $WORKING_DIR

for seed in {1..1}
do
  python ./adr_label_combined.py -P $PROCESSDATADIR/$DATASETFILE -b $WORKING_DIR -s $seed -m $MODEL
done

echo All Done