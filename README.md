### Image-Segmentation-Training-and-Testing

Requirements (all can be eaisly installed using pip or conda):

	1. keras==2.3.1
	2. Tensorflow==1.14.0
	3. opencv-python
	4. numpy
	5. matplotlib
	6. tqdm 

Content Explanation:

	1. all.py file have all the required function needed for doing image segmentation using Vgg based U-Net model.
	 
	2. train.py file trains the model on training images and save the weights for future.
	
	3. predict.py file can do the prediction on sigle as well as whole folder of test images.

Various Outputs:

	1. Checkpionts are saved in checkpoint folder after training.
	
	2. Prediction Ouput images are saved in out folder.

Note:
	 Please pay attention to various comments present in the given scripts for understanding 	the overall work flow.

Reference: 
	Some part of code is taken from some open source community:	
	https://github.com/divamgupta/image-segmentation-keras 
	https://divamgupta.com/image-segmentation/2019/06/06/deep-learning-semantic-segmentation-keras.html
 
