#!/bin/bash
cd "$(dirname "$0")/.."

python slam.py --config configs/scannet/scene0000.yaml
python slam.py --config configs/scannet/scene0054.yaml
python slam.py --config configs/scannet/scene0059.yaml
python slam.py --config configs/scannet/scene0106.yaml
python slam.py --config configs/scannet/scene0169.yaml
python slam.py --config configs/scannet/scene0181.yaml
python slam.py --config configs/scannet/scene0207.yaml
python slam.py --config configs/scannet/scene0233.yaml
