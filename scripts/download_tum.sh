#!/bin/bash

mkdir -p datasets/tum
cd datasets/tum

download() {
    local url=$1
    local seq=$2
    if [ ! -d "$seq" ]; then
        echo "Downloading $seq..."
        wget "$url"
        tar -xzf "$seq.tgz"
        rm "$seq.tgz"
    else
        echo "$seq already exists, skipping."
    fi
}

download "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz" "rgbd_dataset_freiburg1_desk"
download "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk2.tgz" "rgbd_dataset_freiburg1_desk2"
download "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz" "rgbd_dataset_freiburg1_room"
download "https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz" "rgbd_dataset_freiburg2_xyz"
download "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz" "rgbd_dataset_freiburg3_long_office_household"
