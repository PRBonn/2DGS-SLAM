#!/bin/bash
cd "$(dirname "$0")/.."

python slam.py --config configs/replica/office0.yaml
python slam.py --config configs/replica/office1.yaml
python slam.py --config configs/replica/office2.yaml
python slam.py --config configs/replica/office3.yaml
python slam.py --config configs/replica/office4.yaml
python slam.py --config configs/replica/room0.yaml
python slam.py --config configs/replica/room1.yaml
python slam.py --config configs/replica/room2.yaml
