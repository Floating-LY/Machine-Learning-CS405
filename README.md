# Machine-Learning-CS405
此项目为上海交通大学CS405-2机器学习2021年春期第21组大作业代码文件  

小组成员包括：谢泽宇，卓建恒，罗旸  

下为文件说明：  

Comparison.py：此文件使用了segmentation_models.pytorch库（https://github.com/qubvel/segmentation_models.pytorch）  
统一设置ResNet34作为encoder，encoder_depth设置为5,  通过更改模型名字进行训练与验证得到最终结果  
实验结果在报告中Section2部分展示  


Unet.py：此文件定义了1:Unet基础block、Unet上采样模块和Unet网络；2:数据读入Dataset和DataLoader；3:Unet训练的main函数。
参考注释在get_args()函数中可修改超参数和其余设置，linux运行实例："python Unet.py"。
实验结果在报告中Section5部分展示  

cnn.py：此文件定义了1:cnn网络；2:数据读入Dataset和DataLoader；3:cnn训练的main函数。
参考注释在get_args()函数中可修改超参数和其余设置，linux运行实例："python cnn.py"。
实验结果在报告中Section5部分展示  


utils.py：此文件用于计算vinfo和vrand，作为评价模型的指标
