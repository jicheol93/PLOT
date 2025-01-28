# PLOT: Text-based Person Search with Part Slot Attention for Corresponding Part Discovery

![plot](./plot.PNG)

This repository is the official implementation of "[PLOT: Text-based Person Search with Part Slot Attention for Corresponding Part Discovery](https://arxiv.org/abs/2409.13475)" (ECCV 2024)

## Installation

  The code works on 
  - Ubuntu 22.04
  - CUDA 11.7.0
  - NVIDIA RTX A6000
  - Python: 3.10.13
  - PyTorch: 1.13.0

  Conda environment and pytorch installation:
  ```
  conda create -n PLOT python=3.10.13
  conda activate PLOT
  conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
  ```
  

  Additional package installation:
  ```
  pip install -r requirements.txt
  ```

## Data preparation
Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description), ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN) and RSTPReid dataset form [here](https://github.com/NjtechCVLab/RSTPReid-Dataset)

Final `your data root dir` as follows:
```
|-- your data root dir/
|   |-- <CUHK-PEDES>/
|       |-- imgs
|            |-- cam_a
|            |-- cam_b
|            |-- ...
|       |-- reid_raw.json
|
|   |-- <ICFG-PEDES>/
|       |-- imgs
|            |-- test
|            |-- train 
|       |-- ICFG_PEDES.json
|
|   |-- <RSTPReid>/
|       |-- imgs
|       |-- data_captions.json
```

## Training
```
sh run_plot_cuhk.sh
sh run_plot_icfg.sh
sh run_plot_rstp.sh
```

## Evaluation
```
sh eval_plot_cuhk.sh
sh eval_plot_icfg.sh
sh eval_plot_rstp.sh
```
## Download weights
```
```

## Acknowledgments
Parts of our codes are adopted from the following repositories. We sincerely appreciate their contributions.
* https://github.com/openai/CLIP
* https://github.com/anosorae/IRRA/tree/main
* https://github.com/lucidrains/slot-attention
