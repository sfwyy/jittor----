#  Jittor 开放域少样本视觉分类赛题

## 简介

本项目为第四届计图人工智能挑战赛-开放域少样本视觉分类赛题的代码实现，队伍名为**“开始测试”**。本项目的特点：利用小样本微调baseline模型+修改提示词，在B榜取得了top1正确率：0.7168，top5正确率：0.8943的结果。

提示词由ChatGPT对每一类生成更为详细的描述，再对随机选取的图片进行图像增强，之后利用更新后的图像和文本微调基础模型参数。

## 安装

本项目可在1张4090上运行，训练时间约为30分钟（样本增强20分钟+模型微调10分钟）。

#### 运行环境

- ubuntu 20.04 LTS
- python 3.9.19
- jittor 1.3.9.10
- jclip 1.0

#### 安装依赖

正确安装jittor（需要安装最新版）、jclip环境即可，以下是jclip安装命令。

```
pip install ftfy regex tqdm
python setup.py develop
```

#### 预训练模型

预训练模型模型下载地址为[Release ViT-B-32.pkl的权重 · uyzhang/JCLIP (github.com)](https://github.com/uyzhang/JCLIP/releases/tag/权重)，下载后放入项目中`model/`目录下。

## 训练

在训练中包括两部分，第一部分是图像增强，包括旋转、加噪等操作，第二部分是用增强后的图像进行对模型进行微调，一般在12-14轮时效果最好。

`model/`文件夹存的是预训练模型，`saved_models`存的是微调后的模型。

训练可运行以下命令：

```
bash train.sh
```

**注意：**每次选取的四个图像都是随机的，故答案可能有小范围浮动，且没有最终选取的图片list。

## 推理

生成测试集上的结果可以运行以下命令，里面的模型路径和测试数据集路径需修改：

```
python test.py
```

## 最终模型参数量和

利用如下代码对最终模型的参数量进行打印：

```
total_params = sum(p.numel() for p in model.parameters())
print(f"模型的总参数量: {total_params/(1024*1024)}M")
```

得到的结果：

```
模型的总参数量: 144.2749423980713M
```

## 开源链接

热身赛链接：

[llrong/jittor-开始测试-手写数字生成: 第四届计图人工智能挑战赛，队伍“开始测试”热身赛代码及结果，使用jittor框架 | GitLink](https://gitlink.org.cn/p38ij7xsv/jittor_kaishiceshi/tree/master)

赛题1链接：

gitlink: [llrong/jittor-开始测试-开放域少样本视觉分类-main | GitLink](https://www.gitlink.org.cn/p38ij7xsv/jittor_kaishiceshi_jclip/tree/main)

github: [sfwyy/jittor----: 本项目为第四届计图人工智能挑战赛-开放域少样本视觉分类赛题队伍“开始测试”的代码实现。 (github.com)](https://github.com/sfwyy/jittor----)

