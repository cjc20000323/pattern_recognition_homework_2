## 人脸检测

使用的网络是 ResNet-18 + FPN + [CenterNet](https://arxiv.org/abs/1904.07850)的预测头。

### **网络训练**

这次的训练使用的是[WIDER Face](http://shuoyang1213.me/WIDERFACE/)数据集的训练集部分，验证采用其验证集部分。训练的过程如下：

(1) 修改了标注格式以方便处理，修改后的数据可从[链接](https://bhpan.buaa.edu.cn:443/link/1F52E0A94332DD83EA44AA7A78C93AE2)下载；

(2) 修改`det\configs\default.py`中的配置，主要是`_C.DATA.TRAIN_DIR`和`_C.DATA.VAL_DIR`这两个路径（分别指向在第(1)步中修改后数据的训练集和验证集）；

(3) 执行`det\train.py`脚本。

### **扣取人脸图片**

执行`det\crop_face.py`脚本即可使用训练后的网络扣取图片中的人脸，执行脚本前需设置该脚本中`crop_face`函数的参数（主要是`data_path`和`ckpt_path`，分别代表要扣取的图片/图片目录的路径、网络参数检查点的路径）。

目前只考虑`faces96`数据集，因此固定认为每张图片中只有一个人脸。扣取后的图片会按照原先图片目录的结构存放在`crop_face`函数的`save_dir`参数所指向的目录中。

本次扣取时的网络共训练了100 epochs，可从[这里](https://bhpan.buaa.edu.cn:443/link/F06D2F9A4CE3977A92EFE2EDE26877C5)下载。扣取时总共有 3 张图片存在检测失败的情况，对这 3 张图片已进行了手动扣取，扣取后的数据下载地址为：

(1) `det\crop_face.py`脚本的原始输出结果（含3张检测失败的图片）：[链接](https://bhpan.buaa.edu.cn:443/link/B3F8C5CEAD5BCA686D8981E7EB42029B)；

(2) 补正后的结果（3张检测失败的图片已手动扣取）：[链接](https://bhpan.buaa.edu.cn:443/link/1568B2F9AF476380C98A623359EA055D)。
