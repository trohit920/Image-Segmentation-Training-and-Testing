from all import *
import matplotlib.pyplot as plt

################################ Predict Single Image 	######################

out_single = predict(
    inp="Images/images_test/4_resized.png",
    out_fname="Images/out_pre.png",
    checkpoints_path = "Images/checkpoint/vgg_unet_1"
)

plt.imshow(out_single)
plt.show()

################################ Predict Complete test Folder and saving output	######################

out_folder = predict_multiple(
    inp_dir="Images/images_test/",
    out_dir="Images/out/",
    checkpoints_path = "Images/checkpoint/vgg_unet_1"
)

########################## 	Evaluating the model 	#####################

model = vgg_unet(n_classes=200 ,  input_height=416, input_width=608  )

print(model.evaluate_segmentation( inp_images_dir="Images/images_test/"  , annotations_dir="Images/annotation_test/" ) )

