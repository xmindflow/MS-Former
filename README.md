# [MS-Former: Multi-Scale Self-Guided Transformer for Medical Image Segmentation](https://openreview.net/forum?id=pp2raGSU3Wx) (Accepted for oral presentation in MIDL 2023 conference)

![msformer](https://github.com/mindflow-institue/MS-Former/assets/61879630/fe8910b5-b9ed-4cf7-be80-50b8398e13b5)

Multi-scale representations have proven to be a powerful tool since they can take into account both the fine-grained details of objects in an image as well as the broader context. Inspired by this, we propose a novel dual-branch transformer network that operates on two different scales to encode global contextual dependencies while preserving local information. To learn in a self-supervised fashion, our approach considers the semantic dependency that exists between different scales to generate a supervisory signal for inter-scale consistency and also imposes a spatial stability loss within the scale for self-supervised content clustering. While intra-scale and inter-scale consistency losses aim to increase features similarly within the cluster, we propose to include a cross-entropy loss function on top of the clustering score map to effectively model each cluster distribution and increase the decision boundary between clusters. Iteratively our algorithm learns to assign each pixel to a semantically related cluster to produce the segmentation map. Extensive experiments on skin lesion and lung segmentation datasets show the superiority of our method compared to the state-of-the-art (SOTA) approaches. 

## Updates
- Paper accepted in MIDL 2023 for oral presentation

## Installation

```bash
pip install -r requirements.txt
```

## Run Demo
Put your input images in the ```input/image``` folder and just simply run the ```MSFormer.ipynb``` notebook ;)

## Citation
If this code helps with your research, please consider citing the following paper:
</br>

```python
@inproceedings{
  karimijafarbigloo2023msformer,
  title={{MS}-Former: Multi-Scale Self-Guided Transformer for Medical Image Segmentation},
  author={Sanaz Karimijafarbigloo and Reza Azad and Amirhossein Kazerouni and Dorit Merhof},
  booktitle={Medical Imaging with Deep Learning},
  year={2023},
  url={https://openreview.net/forum?id=pp2raGSU3Wx}
}
```
