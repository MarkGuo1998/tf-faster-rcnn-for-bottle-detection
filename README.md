# Tensorflow-Faster-RCNN-for-Bottle-Detection
### 声明与致谢

  - 本文档及本代码的主体部分参考[这个GitHub链接](https://github.com/endernewton/tf-faster-rcnn)。
  - 可以用[labelImg](https://github.com/tzutalin/labelImg )工具处理图片，也可以使用repo中的Matlab代码。这部分代码参见[这个GitHub链接](https://github.com/ruyiweicas/Creat_FRCNN_DataSet)。
  - 可以用两种方式生成图片格式的输出：
	  - 用`./tools/demo.py`批量处理`./data/demo/`中的文件
	  - 把`./tools/test_net.py`的输出`./output/${NET}/${DATASET}/default/*/*.pkl`拷贝到`./data/VOCDevkit2007/VOC2007`中，并使用`draw.py`处理
		  - 具体操作参考[这个链接](https://blog.csdn.net/majinlei121/article/details/78903537)。

### 运行要求

- `Python 3`
- `Tensorflow r1.2`以上
- `cython`， `opencv-python`， `easydict`
- 其他包使用anaconda自带的即可。如果`pandas`报错，请将其降级到`pandas 0.20.3`。

### 如何运行
1. 把这个repo复制到本地

2. 根据GPU型号调整-arch参数
  ```Shell
  cd tf-faster-rcnn/lib
  # Change the GPU architecture (-arch) if necessary
  vim setup.py
  ```

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) / Tesla P100 | sm_37 |

  如果在CPU上运行代码，需要对代码做如下改动：

  - 将`./lib/model/config.py`第270行改为`__C.USE_GPU_NMS = False`。

  - 注释掉`./lib/model/config.py`第12行的 `from nms.gpu_nms import gpu_nms`。

  - 注释掉`./lib/setup.py`第55行的`CUDA = locate_cuda()`；

  	注释掉120行的`Extension('nms.gpu_nms'...`代码。

3. 编译cython模块

  ```Shell
  make clean
  make
  cd ..
  ```

4. 下载[Python COCO API](https://github.com/pdollar/coco)
  ```Shell
  cd data
  git clone https://github.com/pdollar/coco.git
  cd coco/PythonAPI
  make
  cd ../../..
  ```

### 制作符合要求的输入数据

参考[这个GitHub链接](https://github.com/ruyiweicas/Creat_FRCNN_DataSet)。以VOC2007数据集为例。

1. 下载超链接中的repo到`./data/VOCDevkit2007/VOC2007`文件夹（或直接利用我这个repo自带的文件夹）。`VOC2007`下面应当有4个子文件夹，分别是`Annotation`（存储xml文件）、`ImageSets/Main`（存储train、test、validation信息）、`img`（存储原始图片）、`JPEGImages`（生成的图片）。

	另外我们需要在`VOC2007`文件夹下制作一个`output.txt`，格式为

	```
	图片名 物体类别 左上角x坐标 左上角y坐标 右下角x坐标 右下角y坐标
	```

2. 用`Matlab`运行`VOC2007xml.m`，得到xml文件。

3. 用`Matlab`运行`VOC2007txt.m`，得到训练集和测试集的划分信息。

### 训练和测试

1. 将在imagenet上训练好的模型拷贝到`./data/imagenet_weights/`。支持resnet-101和VGG-16两类模型，请确保模型的名称为`res101.ckpt`或`vgg16.ckpt`。

2. 训练（修改config中的参数以改变存储模型的频率）

   ```Shell
   ./experiments/scripts/train_faster_rcnn.sh [GPU_ID] pascal_voc [NET]
   # GPU_ID is the GPU you want to test on
   # NET in {vgg16, res101} is the network arch to use
   # Examples:
   ./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc vgg16
   ```

3. 测试（训练结束后会自动调用一次test）
      ```Shell
      ./experiments/scripts/test_faster_rcnn.sh [GPU_ID] pascal_voc [NET]
      # GPU_ID is the GPU you want to test on
      # NET in {vgg16, res50, res101, res152} is the network arch to use
      # Examples:
      ./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc res101
      ```

4. 检查测试结果（test程序输出的Mean AP似乎有问题）

   - 将输出的*.pkl拷贝到`./data/VOCDevkit2007/VOC2007`中，同时把之前生成的`test.txt`也拷到这个目录下。在该目录下新建`pkl`文件夹。
   - 安装`pickle`模块（py2是`cPickle`）。
   - 运行`draw.py`，到`./pkl`中查看结果。

### 实验结果

1. 在CPU上训练时，每次迭代平均耗时10秒（VGG-16）或20秒（ResNet-101）。

  在GPU（1块Tesla P100）上训练时，每次迭代耗时0.33秒（VGG-16）或0.50秒（ResNet-101）。

  考虑到训练3000步的模型就可以达到相当不错的效果，我们在CPU上最快只需训练10小时，而GPU上最快只需训练30分钟，即可以把得到的模型投入使用。

2. 在CPU上测试时，每张图片平均耗时3秒（VGG-16）或7秒（ResNet-101）。

  在GPU（1块Tesla P100）上测试时，每张图片耗时0.10秒（VGG-16）或0.18秒（ResNet-101）。

  由于任务比较简单，VGG-16和ResNet-101在测试集上的准确率都接近100%。

3. Loss基线：

  - VGG-16：0.12（训到0.16就能用了）
  - ResNet-101：0.35（训到0.40就能用了）

4. Mean AP：

	- 
		（VOC10前的metrics）一般是0.90以上
		
		​	这个数之所以不是1，是评估指标的问题，实际上0.909可以认为是100%识别了
	
	- （VOC10后的metrics）至少0.997

5. 显存消耗：

	- Training 9000MiB
	- Test 7000MiB
		- 
			经实验，4000MiB显存也能带得动，而且速度未见降低太多
		- 2000MiB会让时间增加40%

### 总结与展望

1. 限制faster-rcnn表现的瓶颈主要是网络结构。太简单的网络不足以达到高准确率，太复杂的网络又会显著降低运行速度。在这方面，最好采用GPU进行加速。
2. 我调整了原算法对anchor的选择，原来是3种大小*3种形状=9种anchor，我改成了1种（因为酒瓶口对应的box一定是同种大小的正方形）。修改后未见提速，但是观察到loss下降（不确定是否为实验误差）。
3. 如要再提升效率，可以考虑改进box regression部分的选择策略，使回归的结果也全部都是正方形。
4. ~~目前的算法不能正确处理带瓶盖的瓶子，因为训练时没有获得这部分数据。~~
5. YOLO系虽然很快，但它们不如R-CNN的地方主要是准确率，对这种要求高精度的任务，感觉还是Faster R-CNN更好用一些。如果时间或设备实在达不到要求，以后会考虑YOLO v3。
