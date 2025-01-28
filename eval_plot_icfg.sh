#!/bin/bash
CUDA_VISIBLE_DEVICES=5 \ 
python test_plot.py \
--config_file ./logs/ICFG-PEDES/20240229_normID_TextAggPredWoM2M_Temp0015_5_15_AmpScaler_SharedLocalSlot8TextAttnMask_ReconAllScale001SDMCLScale5_slotLr5_LearnableSlotsIter5/configs.yaml \
#--config_file ./logs/ICFG-PEDES/plot/configs.yaml \
