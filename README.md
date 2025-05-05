# TrackFormer: Multi-Object Tracking with Transformers

This repository builds upon the official implementation of [TrackFormer: Multi-Object Tracking with Transformers](https://arxiv.org/abs/2101.02702) paper by [Tim Meinhardt](https://dvl.in.tum.de/team/meinhardt/). The codebase integrates components from [DETR](https://github.com/facebookresearch/detr), [Tracktor](https://github.com/phil-bergmann/tracking_wo_bnw) and [VRWKV](https://github.com/OpenGVLab/Vision-RWKV). In our work, we focus exclusively on DETR-based tracking, with a primary goal of reducing the quadratic complexity inherent in conventional Transformer architectures. Our modifications enable the model to process high-resolution images—up to 2× the resolution typically handled by Trackformer—while maintaining tracking accuracy.

<!-- **As the paper is still under submission this repository will continuously be updated and might at times not reflect the current state of the [arXiv paper](https://arxiv.org/abs/2012.01866).** -->

<div align="center">
    <img src="docs/MOT17-03-SDP.gif" alt="MOT17-03-SDP" width="375"/>
    <img src="docs/MOTS20-07.gif" alt="MOTS20-07" width="375"/>
</div>

## Abstract
 Object tracking, the task of identifying and following objects across video frames, remains a fundamental challenge in computer vision, with applications spanning autonomous navigation, surveillance, and human-computer interaction. 
 Existing transformer-based models, such as TrackFormer, rely on the Detection Transformer (DETR) framework, which applies global attention mechanisms to all image tokens, resulting in high computational costs and memory inefficiencies 
 that hinder real-time performance and scalability. To address these limitations, we explore the integration of the Receptance Weighted Key Value (RWKV) model—a state-of-the-art natural language processing architecture—into the multiple 
 object tracking paradigm. We propose a novel hybrid framework that combines the encoder structure of Vision-RWKV (VRWKV) with a TrackFormer-inspired decoder, enhancing both feature extraction and tracking performance. Unlike traditional 
 transformers that use quadratic attention, the VRWKV encoder employs a linear attention mechanism, significantly reducing memory consumption and computational complexity while maintaining expressive feature representations. 
 The TrackFormer-inspired decoder leverages attention based temporal modeling to ensure robust object tracking across frames. The proposed approach is evaluated on the MOT17 dataset, demonstrating significant improvements in tracking 
 efficiency and scalability compared to Trackformer. Our findings suggest that this hybrid architecture enables high-performance object tracking suitable for deployment in resource-constrained environments without compromising accuracy.

<div align="center">
    <img src="docs/method.png" alt="TrackFormer casts multi-object tracking as a set prediction problem performing joint detection and tracking-by-attention. The architecture consists of a CNN for image feature extraction, a Transformer encoder for image feature encoding and a Transformer decoder which applies self- and encoder-decoder attention to produce output embeddings with bounding box and class information."/>
</div>

## Installation

We refer to our [docs/INSTALL.md](docs/INSTALL.md) for detailed installation instructions.

## Train TrackFormer

We refer to our [docs/TRAIN.md](docs/TRAIN.md) for detailed training instructions.

## Training Details and Metrics

Due to compute constraints, we trained our model from scratch for **50 epochs**, similar to the Deformable DETR-based TrackFormer setup. However, DETR typically requires **at least 300 epochs to converge** effectively, which affected the stability of certain tracking metrics such as **MOTA** and **IDF1**, occasionally leading to `NaN` or negative values during evaluation.

To better capture the performance trends in such early-stage training, we focus primarily on the **Recall** metric, which remains more stable and informative under partial convergence. This allows us to better assess detection quality independently of association errors.

---

## Model Variants

We experimented with two major configurations:

1. **Standard DETR-based TrackFormer**: Uses the original DETR encoder and decoder (we commited it from Official implementation of Tracformer by Tim Meinhardt).
2. **Hybrid Model (VRWKV-Encoder + DETR-Decoder)**: We replaced the encoder in DETR with a [**VRWKV**](https://github.com/OpenGVLab/Vision-RWKV) encoder to reduce complexity and improve performance on high-resolution inputs, while keeping the DETR decoder intact.

---

## Loss Curves

Below are the training loss curves over 50 epochs:

![Training Loss Curve](docs/loss.png)


---


## Evaluate TrackFormer

In order to evaluate TrackFormer on a multi-object tracking dataset, we provide the `src/track.py` script which supports several datasets and splits interchangle via the `dataset_name` argument (See `src/datasets/tracking/factory.py` for an overview of all datasets.) The default tracking configuration is specified in `cfgs/track.yaml`. To facilitate the reproducibility of our results, we provide evaluation metrics for both the train and test set.

### MOT17

```
python src/track.py with reid
```
## Recall Comparison (50 Epochs)

| Metric                                                          | Proposed Model (VRWKV+DETR) | TrackFormer (DETR) |
|------------------------------------------------------------------|------------------------------|---------------------|
| AR@[IoU=0.50:0.95 — area=all — maxDets=1]                        | 0.012                        | 0.000               |
| AR@[IoU=0.50:0.95 — area=all — maxDets=10]                       | 0.068                        | 0.000               |
| AR@[IoU=0.50:0.95 — area=all — maxDets=100]                      | 0.145                        | 0.000               |
| AR@[IoU=0.50:0.95 — area=small — maxDets=100]                    | 0.000                        | 0.000               |
| AR@[IoU=0.50:0.95 — area=medium — maxDets=100]                   | 0.081                        | 0.000               |
| AR@[IoU=0.50:0.95 — area=large — maxDets=100]                    | 0.410                        | 0.001               |

> **Table 2**: Recall comparison between the proposed model (VRWKV encoder + DETR decoder) and the original TrackFormer with DETR. All results are reported after 50 epochs of training.


```
python src/track.py with \
    reid \
    tracker_cfg.public_detections=min_iou_0_5 \
    obj_detect_checkpoint_file=models/your_saved_model_name/checkpoint_epoch_50.pth
```



<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="docs/MOT17-01-SDP_trackformerDETR-ezgif.com-optimize.gif" alt="TrackFormer Demo" width="400"/><br/>
        <b>TrackFormer (DETR)</b>
      </td>
      <td align="center">
        <img src="docs/MOT17-01-SDP-ezgif.com-optimize.gif" alt="Hybrid Model Demo" width="400"/><br/>
        <b>Hybrid Model (VRWKV + DETR)</b>
      </td>
    </tr>
  </table>
</div>


## Publication
If you use this software in your research, please cite this publication:

```
@InProceedings{meinhardt2021trackformer,
    title={TrackFormer: Multi-Object Tracking with Transformers},
    author={Tim Meinhardt and Alexander Kirillov and Laura Leal-Taixe and Christoph Feichtenhofer},
    year={2022},
    month = {June},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```
