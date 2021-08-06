### 项目介绍
该项目是人脸识别之活体检测的Demo

-----------------------------------------

### 环境配置
```shell
pip install requirements.txt
```
-----------------------------------------

### 训练过程
##### 1. 从视频中提取图像
```shell
python tools/split.video_with_similarity.py
```

##### 2. 从图像中提取人脸作为训练集
```shell
python tools/gen_train_data_with_align.py
```
##### 3. 开始训练
1. 修改configs/config.yaml中的相关参数
2. 生成数据集csv文件
    ```shell
    python gen_train_csv.py
    ```
3. 运行训练脚本
    ```shell
    python train.py
    ```
----------------------------------------
### 推理过程
##### 单例
```shell
python infer/infer.py
```

##### 批量
```shell
python iner/infer_batch.py
```
----------------------------------------
