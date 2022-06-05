# Mid-Term-Project and Final-Term-Project
Code for mid-term and final-term PJ assigned by Prof. Zhang Li, including image classification, object detection and semantic segmentation.. 

## Image Classification
The introduction of image classification lab can be found [here](https://github.com/Evergreen0929/Mid-Term-Project/tree/main/pytorch-cifar-models). Code in `./pytorch-cifar-models`.

Additionally, pretrained weights with different augmentation are provided [here](https://pan.baidu.com/s/1FDBt87OKZ3mL3Y8YcbnnKw), extracting code: `io3w`, including cutout, mixup, cutmix and cutout-MAE pretraining.

## Object Detection
Faster-RCNN and YOLOV3 are both trained and tested on PASCAL VOC. Code in `./easy_detection` and `./YOLOV3` respectively.
Introduction of Faster-RCNN can be found [here](https://github.com/misads/easy_detection/blob/master/_assets/_docs/get_started.md), and YOLOV3 [here](https://github.com/Evergreen0929/Mid-Term-Project/tree/main/YOLOV3).

Object detection models can be found [here](https://pan.baidu.com/s/1EGOhBNv_k0YCE1qE187cBw), extracting code: `uieb`. (Update! models train from scratch and finetuned with Imagenet and Coco pretrained backbone respectively.)

## Semantic Segmentation
Inference with HRNet-w48+OCR pretrained on Cityscapes. Code in `./HR-Net-Semantic-Segmentation`.  
Introduction of HRNet+OCR series can be found [here](https://github.com/Evergreen0929/Mid-Term-Project/tree/main/HR-Net-Semantic-Segmentation).

To inference single image, run: `python tools/inference.py --cfg experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml`
To inference single video, run: `python tools/inference_video.py --cfg experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml`

## Contact
19307140032@fudan.edu.cn
