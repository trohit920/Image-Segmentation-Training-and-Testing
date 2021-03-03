from all import *
import matplotlib.pyplot as plt

model = vgg_unet(n_classes=192 ,  input_height=416, input_width=608  )

model.train(
    train_images =  "Images/images_train/",
    train_annotations = "Images/annotation_train/",
    checkpoints_path = "Images/checkpoint/vgg_unet_1" , epochs=5
)

