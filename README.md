# A Scalable and Generalizable Pathloss Map Prediction

This repo is the official implementation of ["A Scalable and Generalizable Pathloss Map Prediction"](https://arxiv.org/abs/2312.03950) as well as the follow-ups.

## Introduction

**PMNet** (Neural network tailored for Pathloss Map Prediction (PMP)) is described in [arxiv](https://arxiv.org/abs/2103.14030), which capably serves as a backbone for the PMP task.

PMNet achieves strong performance on the PMP task ($10^{-2}$ level RMSE on val), surpassing previous models by a large margin.

<!-- ![overview_PMNet](figure/overview_PMNet.png) -->
<img src="figure/overview_PMNet.jpg" alt="overview_PMNet" width="800" style="float:center" />

<!-- ## Main Results on USC RT Dataset with Pretrained Models -->

## Dataset: Ray-Tracing (RT)-based Channel Measurement (Updating...)

<!-- ![map_USC](figure/map_USC.png)
![bldmap_3D_USC](figure/geometry_USC.png)
![map_UCLA](figure/map_UCLA.png)
![map_Boston](figure/map_Boston.png) -->

<img src="figure/map_USC.png" alt="map_USC" width="300"/> <img src="figure/geometry_USC.png" alt="bldmap_3D_USC" width="300"/> <br/>
<img src="figure/map_UCLA.png" alt="map_UCLA" width="300"/> <img src="figure/map_Boston.png" alt="map_Boston" width="300"/>

**Links for Dataset** <br/>
[USC Dataset](https://drive.google.com/file/d/1uk7no4GwSCQ_sKI3WWnl67llzV1iLQ4l/view?usp=sharing) <br/>
[Radiomapseer Reduced](https://drive.google.com/file/d/12eeypRCnTaCv1WyY9E-FT7ugvdsBTqDN/view?usp=sharing) <br/>
[Radiomapseer Orginal](https://drive.google.com/file/d/1PTaPpLOKraVCRZU_Tzev4D5ZO32tpqMO/view?usp=sharing)

## Available checkpoints for PMNet

| #   | Feature Size | Data-Augmentation | Fine-Tuning    | RMSE     | Download Link                                                                                     |
| --- | ------------ | ----------------- | -------------- | -------- | ------------------------------------------------------------------------------------------------- |
| 1   | 16/H X 16/W  | 4-way flips       | -              | 0.012599 | [Download](https://drive.google.com/file/d/1xxQ8TpS1UdGfh1ELSTAkrlPT5uKIYJei/view?usp=sharing)    |
| 2   | 8/H X 8/W    | 4-way flips       | -              | 0.010570 | [Download](https://drive.google.com/file/d/1SQUdgtFJGRcilAkfDsOsomTzODjjpDsE/view?usp=sharing)    |
| 3   | 16/H X 16/W  | -                 | UCLA Dataset   | 0.031449 | [Download](https://drive.google.com/file/d/1LEtEt7XCMFt7wvChS56XeFi8ySlIxLIJ/view?usp=sharing)    |
| 4   | 16/H X 16/W  | -                 | Boston Dataset | 0.009875 | [Download](https://drive.google.com/file/d/1pwB5C8jHAsu0EQaiU2cJDRMBw4ctpOWI/view?usp=drive_link) |

- #3,4 checkpoints were fine-tuned using (1) which is a pre-trained model with USC Dataset.

### How to use

#### Evaluation

To evaluate above models, refer to the following commands. Or, you can run eval.sh

```bash
python eval.py \
    --data_root [dataset-directory] \
    --network [network-type] \ # pmnet_v1 or pmnet_v3
    --model_to_eval [model-to-eval] \
    --config [config-class-name]
# e.g.,
# python eval.py \
#    --data_root '/USC/' \
#    --network 'pmnet_v3' \
#    --model_to_eval 'config_USC_pmnetV3_V2_epoch30/16_0.0001_0.5_10/model_0.00012.pt' \
#    --config 'config_USC_pmnetV3_V2'
```

#### Train

To train PMNet, please refer to train.sh

```bash
python train.py -d [dataset-root] -n [network-type] -c [config-class-name]
# e.g., python train.py -d '/USC/' -n 'pmnet_v3' -c 'config_USC_pmnetV3_V2'
```

## Citation

```

@inproceedings{lee2023pmnet,
title={PMNet: Robust Pathloss Map Prediction via Supervised Learning},
author={Ju-Hyung Lee and Omer Gokalp Serbetci and Dheeraj Panneer Selvam and Andreas F. Molisch},
year={2023},
month={December},
booktitle={Proceedings of IEEE Global Communicaions Conference (GLOBECOM)},
}

```

```

```
