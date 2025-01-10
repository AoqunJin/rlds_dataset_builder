#!/bin/bash

# Defining a Task List
task_suites=("libero_spatial" "libero_object" "libero_goal" "libero_10" "libero_90")
raw_data_dir="/data/xtydata/libero/datasets"
target_dir_suffix="_no_noops"

# Iterate through the list of tasks and execute commands
for task_suite in "${task_suites[@]}"; do
    raw_dir="${raw_data_dir}/${task_suite}"
    target_dir="${raw_data_dir}/${task_suite}${target_dir_suffix}"
    echo "Executing: python regenerate_libero_dataset.py \
        --libero_task_suite ${task_suite} \
        --libero_raw_data_dir ${raw_dir} \
        --libero_target_dir ${target_dir}"
    
    python regenerate_libero_dataset.py \
        --libero_task_suite "${task_suite}" \
        --libero_raw_data_dir "${raw_dir}" \
        --libero_target_dir "${target_dir}"
done
