# Implementing-Deep-Learning-in-the-field-of-Radiology-and-Health-Care-using-Python
In my internship at “RAD365” (https://www.rad365.com/), I was design various classiﬁcation and segmentation CNN models using Python Keras, Opencv, Sklearn etc. libraries. Here I implement CNN models for CT, MRI, X-RAY images.

# libraries used in this project:
1. Numpy -- for matrix and array related operations
2. Matplotlib -- for plotting
3. Scikit image - for image read write,specificly I used it for .nii image read
4. pydicom -- for dicom image array read
5. Opencv -- for image processings opperation like blurring,thresholding etc etc.
6. Keras -- for Deep learning architecture building

# Work

1. The first project is a multiclass classification problem, where I develop a Transfer learning based CNN model for four classes (1 for meningioma, 2 for glioma, 3 for pituitary    tumour 4 for Normal). [BRATS 2015 dataset]

## Deep learning architecture I used during this project:
![image](https://user-images.githubusercontent.com/33135767/92586289-ac2e3a80-f2b3-11ea-8cf0-5438a054e481.png)

![Multiclass brain tumor classi loss](https://user-images.githubusercontent.com/33135767/92586083-70937080-f2b3-11ea-8d54-714c26596e4c.png) 
<img src="https://user-images.githubusercontent.com/33135767/92586092-72f5ca80-f2b3-11ea-8f79-2b809b4f2f83.png" width="800" height="400" />
   
2. The second project is a binary segmentation problem, where I develop an Encoder-Decoder based CNN model for segmenting spinal cord from MRI image.

## Deep learning architecture I used during this project:

![image](https://user-images.githubusercontent.com/33135767/92586254-a46e9600-f2b3-11ea-8b24-bb838960dd90.png)

3. The third project is a segmentation problem, where I develop an Encoder-Decoder based CNN model for four classes (1 for meningioma, 2 for glioma, 3 for pituitary tumour 4 for    Normal) segmentation. [BRATS 2015 dataset]

## Deep learning architecture I used during this project:

![image](https://user-images.githubusercontent.com/33135767/92586254-a46e9600-f2b3-11ea-8b24-bb838960dd90.png)

![predicted mask spinal cord](https://user-images.githubusercontent.com/33135767/92586088-71c49d80-f2b3-11ea-9101-95a385801e11.png)
![predicted mask](https://user-images.githubusercontent.com/33135767/92586091-725d3400-f2b3-11ea-868e-d5648b6b31c3.png)
![Spinal cord segmentation network IOU value](https://user-images.githubusercontent.com/33135767/92586094-738e6100-f2b3-11ea-9468-b71b6c622f63.png)
![Spinal cord segmentation network loss](https://user-images.githubusercontent.com/33135767/92586095-738e6100-f2b3-11ea-9bc2-003ca044c901.png)
![actual brain image t1c](https://user-images.githubusercontent.com/33135767/92586098-7426f780-f2b3-11ea-89d7-95145325f813.png)
![actual brain mask](https://user-images.githubusercontent.com/33135767/92586100-74bf8e00-f2b3-11ea-8b50-7f191e408e78.png)
![actual spinal cord image](https://user-images.githubusercontent.com/33135767/92586103-74bf8e00-f2b3-11ea-87fa-6dd5e656b215.png)
![actual spinal cord mask](https://user-images.githubusercontent.com/33135767/92586104-75582480-f2b3-11ea-8b01-4424c3d8ede2.png)
![binary segmentation actual image](https://user-images.githubusercontent.com/33135767/92586105-75f0bb00-f2b3-11ea-9579-247c5ee86bd7.png)
![binary segmentation actual mask](https://user-images.githubusercontent.com/33135767/92586109-76895180-f2b3-11ea-91d1-2923463bcc94.png)
![binary segmentation predicted mask](https://user-images.githubusercontent.com/33135767/92586110-76895180-f2b3-11ea-8359-5cb1fee64728.png)
![Multi class Brain tumor segmenation architecture iou value](https://user-images.githubusercontent.com/33135767/92586112-7721e800-f2b3-11ea-9987-0f124216ff68.png)
![Multi class Brain tumor segmenation architecture loss](https://user-images.githubusercontent.com/33135767/92586115-77ba7e80-f2b3-11ea-92f2-df45201d7d34.png)
![multi_classification_accuracy](https://user-images.githubusercontent.com/33135767/92586117-77ba7e80-f2b3-11ea-8104-19dc75670035.png)




[BRATS 2015 dataset]: https://sites.google.com/site/braintumorsegmentation/home/brats2015
