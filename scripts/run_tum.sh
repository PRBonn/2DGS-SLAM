#!/bin/bash
cd "$(dirname "$0")/.."

python slam.py --config configs/tum/fr1_desk.yaml
python slam.py --config configs/tum/fr1_desk2.yaml
python slam.py --config configs/tum/fr1_room.yaml
python slam.py --config configs/tum/fr2_xyz.yaml
python slam.py --config configs/tum/fr3_office.yaml
