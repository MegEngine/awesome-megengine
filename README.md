# Awesome MegEngine

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

* [MegEngine_CU11](https://github.com/Qsingle/MegEngine_CU11): 包含 CUDA11 支持的 MegEngine Python Wheel 包
* [MgeEditing](https://github.com/Feynman1999/MgeEditing)：基于 MegEngine 的图像库
* [Echo](https://github.com/digantamisra98/Echo)：一个优秀的算子库
* [ClearML(Allegro Trains)](https://github.com/allegroai/clearml): ML/DL 开发和生产套件，包含 Experiment Manager / ML-Ops / Data-Management 三大核心模块
* [MegEngine.js](https://www.npmjs.com/package/megenginejs)：MegEngine javascript 版本，可以在 javascript 环境中快速部署 MegEngine 模型
* [commitlint config megengine](https://www.npmjs.com/package/commitlint-config-megengine?activeTab=readme)：MegEngine commitlint 配置项
* [Cascade RCNN ](https://github.com/Antinomy20001/Megvii-AI-Traffic-Sign-Detection-Open-Source-Competition---Unofficial-2nd-Place-Solution)：一种 Cascade RCNN 实现
* [Megvision](https://github.com/Qsingle/Megvision)：一些经典 CV 模型的实现和权重
* [Models](https://github.com/MegEngine/Models)：MegEngine 实例训练代码，包含各类任务的基本实现，是初学者最好的参考物
* [MgeConvert](https://github.com/MegEngine/mgeconvert)：MegEngine Traced module 模型格式转换器，支持 ONNX / TFLite / Caffe 等各类导出格式
* [MegFile](https://github.com/megvii-research/megfile)：一个可以完美抽象 S3、HTTP、本地文件等协议的 python 文件接口库，是 smart-open 库的升级版
* [MegFlow](https://github.com/MegEngine/MegFlow)：面向计算机视觉应用的流式计算框架，提供了一套可快速完成 AI 应用部署的视觉解析服务方案
* [MegSpot](https://github.com/MegEngine/MegSpot)：一款提供免费免登录、高效、专业、跨平台的图片&视频的对比的 PC 应用工具
* [BaseCls](https://github.com/megvii-research/basecls)：极其全面的分类模型库，提供了海量模型的训练代码和预训练权重，全部算法均可以快速部署到硬件上
* [Swin Transformer](https://github.com/MegEngine/swin-transformer)：Swin Transformer 的实现，支持 DTR 功能减少显存占用量
* [MegFLow-Cat_Feeder](https://github.com/rcxxx/MegFlow/tree/master/flow-python/examples/cat_feeders)：基于 `MegFlow` 框架的猫咪检测以及自动投喂的解决方案
* [MegPeak](https://github.com/MegEngine/MegPeak)：处理器性能测试工具
* [Netron]( https://netron.app/) 已可视化 MegEngine TracedModule，欢迎使用 [模型示例文件](https://data.megengine.org.cn/models/traced_shufflenet.tm) 体验
* [MegCC](https://github.com/MegEngine/MegCC)：一个运行时超轻量，高效，移植简单的深度学习模型编译器
* MegEngine 原创论文复现：
    * [RepLKNet](https://github.com/MegEngine/RepLKNet)：[CVPR 2022] Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs
    * [GyroFlow](https://github.com/MegEngine/GyroFlow): [ICCV 2021] GyroFlow: Gyroscope-Guided Unsupervised Optical Flow Learning
    * [ICD](https://github.com/MegEngine/ICD): [NeurIPS 2021] Instance-Conditional Knowledge Distillation for Object Detection
    * [FINet](https://github.com/MegEngine/FINet): [AAAI 2022] FINet: Dual Branches Feature Interaction for Partial-to-Partial Point Cloud Registration
    * [OMNet](https://github.com/MegEngine/OMNet): [ICCV 2021] OMNet: Learning Overlapping Mask for Partial-to-Partial Point Cloud Registration
    * [CREStereo](https://github.com/megvii-research/CREStereo): [CVPR 2022] Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation
    * [D2C-SR](https://github.com/megvii-research/D2C-SR): [ECCV 2022] D2C-SR: A Divergence to Convergence Approach for Real-World Image Super-Resolution
    * [FST-Matching](https://github.com/megvii-research/FST-Matching): [ECCV 2022] Explaining Deepfake Detection by Analysing Image Matching
    * [zipfls](https://github.com/megvii-research/zipfls): [ECCV 2022] Zipf's LS: Efficient One Pass Self-distillation with Zipf's Label Smoothing
    * [HDR-Transformer](https://github.com/megvii-research/HDR-Transformer): [ECCV 2022] Ghost-free High Dynamic Range Imaging with Context-aware Transformer
* 经典论文模型对应结构的 MegEngine inference 函数：
    * [Masked Autoencoders Are Scalable Vision Learners](https://github.com/Asthestarsfalll/MAE-MegEngine)
    * [Wide Residual Networks](https://github.com/zhaoqyu/WRN-MegEngine )
    * [ResNeSt: Split-Attention Networks](https://github.com/Asthestarsfalll/ResNeSt-MegEngine)
    * [Visual Attention Network](https://github.com/Asthestarsfalll/VAN-MegEngine)
    * [A ConvNet for the 2020s](https://github.com/Asthestarsfalll/ConvNeXt-MegEngine)
    * [UniFormer: Unifying Convolution and Self-attention for Visual Recognition](https://github.com/Asthestarsfalll/UniFormer-MegEngine)
    * [Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection](https://github.com/CV51GO/GFLv2_Megengine)
    * [Probabilistic Anchor Assignment with IoU Prediction for Object Detection](https://github.com/CV51GO/PAA_Megengine)
    * [OTA: Optimal Transport Assignment for Object Detection](https://github.com/CV51GO/OTA_Megengine)
    * [Pyramid Scene Parsing Network](https://github.com/Asthestarsfalll/PSPNet-MegEngine)
    * [GhostNet: More Features from Cheap Operations](https://github.com/CV51GO/GhostNet_Megengine)
    * [Squeeze-and-Excitation Networks ](https://github.com/Asthestarsfalll/SENet-MegEngine)
    * [OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://github.com/zhaoqyu/openpose-mge-pt)
    * [Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net](https://github.com/Asthestarsfalll/IBNNet-MegEngine)
    * [HarDNet: A Low Memory Traffic Network](https://github.com/CV51GO/HarDNet_Megengine)
    * [Learning to Navigate for Fine-grained Classification](https://github.com/wwhio/megmodels)
    * [FishNet: a versatile backbone for image, region, and pixel level prediction](https://github.com/CV51GO/FishNet_Megengine)
    * [A Light CNN for Deep Face Representation with Noisy Labels](https://github.com/Asthestarsfalll/LightCNN-MegEngine)
    * [Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](https://github.com/wwhio/megmodels)
    * [Towards Compact Single Image Super-Resolution via Contrastive Self-distillation](https://github.com/wwhio/megmodels)
    * [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://github.com/wwhio/megmodels)
    * [Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images](https://github.com/wwhio/megmodels)
    * [Densely Connected Convolutional Networks](https://github.com/Asthestarsfalll/DenseNet-MegEngine)
    * [Aggregated Residual Transformations for Deep Neural Networks](https://github.com/zhaoqyu/ResNeXt-MGE)
    * [Learning Transferable Visual Models From Natural Language Supervision](https://github.com/Asthestarsfalll/CLIP-MegEngine)
    * [Image Super-Resolution Using Very Deep Residual Channel Attention Networks](https://github.com/wwhio/megmodels)
    * [Residual Dense Network for Image Super-Resolution](https://github.com/wwhio/megmodels.git)
    * [BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation](https://github.com/Asthestarsfalll/BiSeNet-MegEngine)
