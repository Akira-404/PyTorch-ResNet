# PyTorch-ResNet
基于PyTorch的ResNet

**测试环境：Ubuntu-20.04，1050Ti**  

## 使用方法：

提前下好官方提供的预训练模型并更改文件名如下 

```python
model_weight_path = './resnet34-pre.pth'
```

运行train.py进行训练

## 文件说明  

**model**.py:ResNet模型  

**train_net**.py:训练脚本  

**test_net**.py:测试脚本    

**class_indices**.json:自动生成的目标索引文件  

**data_perparation**.py:用于分割训练集和测试集  

**my_dataset**.py:继承DataSet类，用于实现DataLoader  

**.pth**:权重文件  