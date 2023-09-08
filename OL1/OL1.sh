#!/bin/sh


# Run Local
./OL1.py wdbc adam 0
./OL1.py wdbc adam 1
./OL1.py wdbc rmsprop 0
./OL1.py wdbc rmsprop 1
./OL1.py wdbc sgd 0
./OL1.py wdbc sgd 1
./OL1.py iris adam 0
./OL1.py iris adam 1
./OL1.py iris rmsprop 0
./OL1.py iris rmsprop 1
./OL1.py iris sgd 0
./OL1.py iris sgd 1


#singularity exec /home/shared/sif/csci-2023-Fall.sif python3 OL1.py
# or
#apptainer exec /home/shared/sif/csci-2023-Fall.sif python3 OL1.py
