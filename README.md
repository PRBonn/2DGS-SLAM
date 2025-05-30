<p align="center">
  <h1 align="center">Globally Consistent RGB-D SLAM with 2D Gaussian Splatting</h1>

  <p align="center">
    <a href="https://github.com/PRBonn/2DGS-SLAM"><img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54" /></a>
    <a href="https://github.com/PRBonn/2DGS-SLAM"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
    <img src="https://img.shields.io/badge/Paper-pdf-blue.svg?style=flat-square" />
    <a href="https://lbesson.mit-license.org/"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" /></a>
  </p>
  
  <p align="center">
    <a href="https://www.ipb.uni-bonn.de/people/xingguang-zhong/index.html"><strong>Xingguang Zhong</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/yue-pan/index.html"><strong>Yue Pan</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/liren-jin/index.html"><strong>Liren Jin</strong></a>
    ·
    <a href="https://www.tudelft.nl/en/staff/m.popovic/?cHash=07e8a5fb4eda6d511853b2bacaa92260"><strong>Marija Popović</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/jens-behley/"><strong>Jens Behley</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/cyrill-stachniss/"><strong>Cyrill Stachniss</strong></a>
  </p>

  <h3 align="center">
    <a href="#">Paper</a> |
    <a href="#">Video</a>
  </h3>
</p>

<p align="center">
  <img src="media/overview.jpg" alt="Teaser Image" />
</p>

Tracking & Mapping | Map State Update |
:-: | :-: |
<video src='https://github.com/user-attachments/assets/2ac63383-3281-4231-a10f-2fc13d8bf1de.pm4'> | <video src='https://github.com/user-attachments/assets/6d33b9be-1711-4060-8782-b75622fc169f.mp4'> |

## Abstract
<details>
  <summary><strong>[Click to expand]</strong></summary>

Recently, 3D Gaussian splatting-based RGB-D SLAM has demonstrated remarkable performance in high-fidelity 3D reconstruction. However, the lack of depth rendering consistency and efficient loop closure hinders the geometric quality and global consistency of the reconstructed maps in online settings. In this paper, we present **2DGS-SLAM**, an RGB-D SLAM system that leverages 2D Gaussian splatting as the map representation. By exploiting the depth-consistent rendering property of the 2D variant, we propose an accurate and robust camera pose optimization method, achieving geometrically faithful 3D reconstruction. Furthermore, we integrate efficient loop detection and camera relocalization by leveraging MASt3R, a 3D foundation model, and enable real-time map updates through a locally maintained active map. Experimental results show that our 2DGS-SLAM achieves superior tracking accuracy, higher surface reconstruction quality, and improved global consistency compared to existing rendering-based SLAM methods, while also maintaining high-fidelity image rendering and enhanced computational efficiency.

</details>

## Contact
If you have any questions, Feel free to contact:
- Xingguang Zhong {[zhong@igg.uni-bonn.de]()}
- Yue Pan {[yue.pan@igg.uni-bonn.de]()}
- Liren Jin {[ljin@uni-bonn.de]()}
