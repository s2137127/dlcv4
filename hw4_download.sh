#!/bin/bash
cd ./byol-pytorch
wget -O fine_best_C41.pth https://www.dropbox.com/s/igp366t73hcnhrz/fine_best_C41.pth?dl=1
cd ..
cd ./DirectVoxGO
wget -O fine_last.tar https://www.dropbox.com/s/iwp4rb8e1w6yvx6/fine_last.tar?dl=1
cd ./logs/nerf_synthetic/dvgo_hotdog
wget -O coarse_last.tar https://www.dropbox.com/s/jt686ihvs4j7k2g/coarse_last.tar?dl=1
