# nn~~W~~Net: Rethinking the Use of Transformers in Biomedical Image Segmentation and Calling for a Unified Evaluation Benchmark

This is the official code of [nn~~W~~Net: Rethinking the Use of Transformers in Biomedical Image Segmentation and Calling for a Unified Evaluation Benchmark](https://) (CVPR2025).

## How to Use
- Download and configure [**nnUNet**](https://github.com/MIC-DKFZ/nnUNet)

- Move **nnUNetTrainer_WNet2D.py** to **.../nnUNet/nnunetv2/training/nnUNetTrainer/** of the configured nnUNet

- Use nn~~W~~Net just like nnUNet:

> Data Preprocessing
```
nnUNetv2_plan_and_preprocess -d 11 --verify_dataset_integrity
```

> Training
```
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 2d 0 -tr nnUNetTrainer_WNet2D
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 100 2d 1 -tr nnUNetTrainer_WNet2D
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 100 2d 2 -tr nnUNetTrainer_WNet2D
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 100 2d 3 -tr nnUNetTrainer_WNet2D
CUDA_VISIBLE_DEVICES=4 nnUNetv2_train 100 2d 4 -tr nnUNetTrainer_WNet2D
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 66 3d_lowres 0 -tr nnUNetTrainer_WNet3D
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 66 3d_lowres 0 -tr nnUNetTrainer_WNet3D_L
```

> Best Configuration
```
nnUNetv2_find_best_configuration 100 -c 2d -tr nnUNetTrainer_WNet2D
nnUNetv2_find_best_configuration 66 -c 3d_lowres -tr nnUNetTrainer_WNet3D
```

> Testing
```
nnUNetv2_predict -i .../nnUNetFrame/nnUNet_raw/Dataset100_your_dataset/imagesTs/ -o .../your_predict_path/ -d 100 -c 2d -tr nnUNetTrainer_WNet2D
```


## Discussion on Encoder-Decoder Architectures

- **Summary and Abstraction**
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/nnWNet/blob/main/figure/Summary%20of%20three%20architectures%20in%20biomedical%20image%20segmentation.png" width="100%" >
</p>
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/nnWNet/blob/main/figure/Demonstration%20of%20various%20architectures%20along%20with%20their%20corresponding%20models.png" width="100%" >
</p>

> Three main architectures: **encoder-decoder**, **encoder-bottleneck-decoder**, and **encoder-bridge-decoder**.

> Four component modules: **encoder**, **decoder**, **bottleneck**, and **bridge**.

> Design of component modules: the **sequential connection** of pure convolutional layers or pure transformer layers, the **series or parallel connection** of convolutional layers and transformer layers. 

> **Convolution** has two key characteristics: local connections and parameter sharing. Local connections ensure that the extracted features are specific to the local input, while parameter sharing makes the features translationally invariant. This design enhances computational efficiency while enabling the convolution to focus on local details.

> The sequence-to-sequence **transformer** utilizes a global self-attention mechanism to capture long range dependencies and global information.

- **Design Contradiction**

> The current combination method forces the transformer layer to use local features as input to extract long-range dependencies, and enforces the convolutional layer to operate on global features to extract locally-focused features. Global features and local features are generated alternately and cannot be transmitted continuously and stably throughout the model.

## Integrating Transformers without Contradictions
- **Overview**
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/nnWNet/blob/main/figure/Overview.png" width="100%" >
</p>
   
- **Address the Contradiction**
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/nnWNet/blob/main/figure/Transmission%20of%20global%20and%20local%20features.png" width="60%" >
</p>

> The global and local features in ~~W~~Net can flow continuously throughout the model and exchange information with each other at each scale, which not only addresses the contradiction, but also effectively fuses the global and local features into a unified representation.

- **Effective Receptive Field**
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/nnWNet/blob/main/figure/Effective%20ReceptiveFields.png" width="100%" >
<br>Effective receptive fields (ERFs) of LSBs and GSBs on ISIC2017, Kvasir-SEG, CREMI (average over 100 images). Top row: The ERFs of residual blocks of LSBs in the second encoder. Middle row: The ERFs of residual blocks of LSBs in the first decoder. Botton row: The ERFs of 11×11 depth-wise convolution self-attentions of GSBs between the first decoder and the second encoder. (a) Scale 1. (b) Scale 2. (c) Scale 3. (d) Scale 4.
</p>

> As the scale increases, the effective receptive fields (ERFs) of both LSBs and GSBs expand. At the same scale, the ERF of LSB is significantly smaller than that of GSB. Notably, at scale 4, the ERFs of the two become complementary, with regions in LSB appearing redder, while the corresponding regions in GSB appear bluer. This is because LSBs are composed of convolutions and focus more on local details, whereas GSBs are composed of transformers and tend to capture long-range dependencies.

- **Motivation of Architecture Design**
> In practical large-scale applications, we find that the two stage cascaded UNet can further improve performance. Therefore, we try to expand the U-shaped architecture to W-shape to simulate and achieve the cascade effect. In order to capture long-range dependencies while focusing on local details, we introduce Global Scope Bridges (GSBs) at each scale to achieve a continuous and stable flow of local and global features. Because the W-shaped architecture directly transmits high-resolution features between input and output and maintains multi-scale features of low, medium, and high resolutions, it performs better.

## Calling for a Unified Evaluation Benchmark
> The current biomedical image segmentation models lack a unified evaluation benchmark. Different studies have significant discrepancies in experimental datasets, image preprocessing strategies (such as resampling, region of interest cropping), training and validation set divisions, evaluation metrics, and some key hyperparameters (such as patch size, loss function, and number of training epochs). This lack of standardization makes it challenging to compare results across studies and evaluate the true performance of different models. Some models may excel in specific datasets, but fail to generalize in others. Moreover, certain models that claim state-of-the-art performance may not perform well under a unified benchmark and even worse than a well-designed UNet.

> nnUNet is an automatically configured segmentation framework that sets image preprocessing, data augmentation, and training hyperparameters. It not only serves as an out-of-the-box segmentation solution but also provides a unified benchmark for evaluating model performance. We conduct a comprehensive and fair comparison based on the nnUNet framework.

## Quantitative Comparison
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/nnWNet/blob/main/figure/Comparisonwithstate-of-the-artmodelson2Dand3Ddatasets.png" width="100%" >
</p>


## Qualitative Comparison
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/nnWNet/blob/main/figure/Qualitative%20results%20of%20different%20models%20on%202D%20datasets.png" width="100%" >
<br>(a) Raw images. (b) Ground truth. (c) TransAttUNet. (d) nnUNet. (e) nnWNet.
</p>
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/nnWNet/blob/main/figure/Qualitative%20results%20of%20different%20models%20on%203D%20datasets.png" width="100%" >
<br>(a) Raw images. (b) Ground truth. (c) CoTr. (d) nnUNet. (e) nnWNet.
</p>

## Citation
>If our work is useful for your research, please cite our paper:
```
```
