#!/bin/bash

NREPS=4

parallel --jobs 4 --keep-order --plus --progress "
    mkdir -p jobs/job-{0#} &&
    cp scenario.qmd jobs/job-{0#} &&
    quarto render jobs/job-{0#}/scenario.qmd -P job:{#} &&
    rm jobs/job-{0#}/scenario.qmd" ::: $(seq $NREPS)

quarto render viz.qmd
