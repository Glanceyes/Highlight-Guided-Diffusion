# Highlight-Guided Diffusion



Official implementation of **Highlight-Guided Diffusion: Scribble can be Good Guidance**

[![Project page](https://img.shields.io/badge/Website-warming_up-red)](https://highlight-guided.io)
 [![Paper](https://img.shields.io/badge/Paper-warming_up-red)](#) 

<br/>



![Highlight-Guided-Diffusion-Demo](assets/Highlight-Guided-Diffusion-Demo.gif)

<br/>

<br/>

## Abstract



Diffusion models, particularly those like Stable Diffusion, have achieved notable success in text-to-image synthesis. Despite their success, aligning generated images with precise user intent remains a challenge, largely due to the abstract nature of text and its lack of spatial information. Traditional methods like bounding boxes and masks, used to guide the diffusion process, present their own limitations: bounding boxes fail to capture the complete shape and orientation of objects, whereas masks entail a high annotation cost.

In this research, we introduce 'Highlight-Guided Diffusion', a novel approach in the realm of scribble-based diffusion generation without additional fine-tuning. Our research focues on redefining 'scribble', drawing inspiration from the concept used in weakly supervised semantic segmentation. We redefine a scribble as a 'highlight', a simple stroke that not only indicates the object's position and size but also conveys its abstract shape and direction. This redefinition aims to balance the spatial precision offered by masks with the simplicity of bounding boxes, facilitating a more intuitive and user-friendly approach to image synthesis.

Evaluations on the PASCAL-Scribble Dataset demonstrate the capability of our method to produce high-fidelity images closely aligned with user-provided highlights. In addition, we provide an accessible web application and a live demo that underscore this model's real-world applicability and user-friendliness.

<br/>



##  To Do


- [ ] Upload codes
- [ ] Release a paper



<br/>


## Method


![Method](assets/Method.jpeg)

<br/>


## Results



![Figure1](assets/Figure1.jpeg)

![Figure2](assets/Figure2.jpeg)



![Figure3](assets/Figure3.jpeg)

<br/>

<br/>



## Related Works



[BoxDiff: Text-to-Image Synthesis with Training-Free Box-Constrained Diffusion](https://github.com/showlab/BoxDiff)

[Dense Text-to-Image Generation with Attention Modulation](https://github.com/naver-ai/DenseDiffusion)

[Generative Data Augmentation Improves Scribble-supervised Semantic Segmentation](https://arxiv.org/abs/2311.17121v1)

[Grounded Text-to-Image Synthesis with Attention Refocusing](https://github.com/Attention-Refocusing/attention-refocusing)