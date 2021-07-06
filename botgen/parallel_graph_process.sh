#!/bin/bash

FILEPATH="/p/adversarialml/as9rw/datasets/raw_botnet/novA.txt"

for i in {0..10}
do
    python create_graphs.py --id ${i} --filepath $1 --n_splits 10 &
done

wait
echo "Generated all graphs!"