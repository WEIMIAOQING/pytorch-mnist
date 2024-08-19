<h1>MINIST Classification</h1>
该项目是基于pytorch的mnist图像分类任务，其中lenet、vgg、ResNet18和transformer文件夹是分别采用lenet5、vgg16、ResNet18和vision transformer模型进行mnist数据集分类<br>   
<h2>lenet5模型进行mnist数据集分类</h2>
lenet5的网络结构一共7层，分别是输入层、卷积层、池化层、卷积层、池化层、全连接层和输出层。lenet5网络结构的代码在该文件夹中的net.py文件中。
模型的训练和测试文件见该文件夹中的train.py文件。<br>该模型结构简单，易于实现，能比较快速精确的进行mnist手写数据集的识别，能在几个周期就能得到98%的精度。
<h2>vgg16模型进行mnist数据集分类</h2>
vgg16一共有16层，包括13个卷积层和3个全连接层。先经过64个卷积核的两次卷积后，进行最大池化，再经过两次128个卷积核卷积后，进行最大池化；接着经过3次256个卷积核卷积后，采用最大池化，再重复两次三个512个卷积核卷积后，再最大池化，最后经过三次全连接。
vgg16网络结构代码在在该文件夹中的net.py文件中。模型的训练和测试文件见该文件夹中的train.py文件。<br>
但该模型由于网络深度的增加在一定程度上影响了网络最终的性能，出现了梯度消失问题，并且参数量大，计算复杂度高，训练时间长。
<h2>ResNet18模型进行mnist数据集分类</h2>
ResNet18是基于残差块连接的卷积神经网络，该网络结构的代码在该文件夹中的net.py文件中。模型的训练和测试文件见该文件夹中的train.py文件。<br>
采用残差块可以训练更深残差块可以缓解梯度消失问题，使网络能够训练更深的结构而不会显著增加训练难度，mnist手写数据集的识别精度98%左右。
<h2>vision transformer模型进行mnist数据集分类</h2>
vision transformer是基于 Transformer 架构，将图像分割成多个 patch 并将其视为序列处理。该网络结构代码在在该文件夹中的net.py文件中。模型的训练和测试文件见该文件夹中的train.py文件。<br>
该模型在mnist上也达到了98%左右的精度。