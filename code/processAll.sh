#!/bin/bash


if [ $# -lt 2 ]
then
	echo "Apply all recognition step to all inkml files and produce the LG files"
	echo "Copyright (c) H. Mouchere, 2018"
	echo ""
	echo "Usage: processAll <input_inkml_dir> <output_lg_dir>"
	echo ""
	exit 1
fi

INDIR=$1
OUTDIR=$2

if ! [ -d $OUTDIR ] 
then
	mkdir $OUTDIR
fi

for file in $1/*.inkml
do
    echo "Recognize: $file"
    BNAME=`basename $file .inkml`
    OUT="$OUTDIR/$BNAME.lg"
    python3 ./segmenter.py -i $file -o $OUT 
    ERR=$? # test de l'erreur au cas où...
     python3 ./segmentSelect.py -o $OUT  $file $OUT 
    ERR=$ERR || $?
     python3 ./symbolReco.py  -o $OUT $file $OUT
    ERR=$ERR || $?
     python3 ./selectBestSeg.py -o $OUT $OUT 
    ERR=$ERR || $?
    
    if [ $ERR -ne 0 ]
    then 
        echo "erreur !" $ERR
        exit 2
    fi
done
echo "done."
