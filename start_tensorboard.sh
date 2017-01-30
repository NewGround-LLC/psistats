#!/bin/bash
#
# This is uitlity script to start Tensorboard visualisation tool
#

if ( ! getopts "l:" opt); then
        echo "Usage: `basename $0` options (-l the training data directory)";
        exit $E_OPTERROR;
fi

logdir=
while getopts ":l:" opt; do
  case $opt in
    l)
      	logdir=$OPTARG
      	;;
    \?)
      	echo "Invalid option: -$OPTARG, please use -l <path to train data directory>" >&2
      	exit 1
      	;;
    :)
	echo "Option -$OPTARG requires an argument." >&2
	exit 1
	;;	
  esac
done

echo $logdir

/usr/local/bin/python -m tensorflow.tensorboard --logdir=$logdir
