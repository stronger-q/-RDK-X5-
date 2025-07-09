##整体概述

![c9b72868-2b04-4fe1-8b19-96013e78adce](file:///D:/Typedown/c9b72868-2b04-4fe1-8b19-96013e78adce.png)







## 在RDK x5上部署MAVROS2

    sudo apt-get install ros-foxy-mavros 
    sudo apt-get install ros-foxy-mavros-extras
    
    运行install_geographiclib_datasets.sh
    
    git clone -b ros2 https://github.com/mavlink/mavros.git
    cd mavros/mavros/scripts
    sudo ./install_geographiclib_datasets.sh
    
    
    最后一句sudo ./install_geographiclib_datasets.sh需要根据实际网络环境等待一会儿，运行完终端显示如下
    
    ubuntu@ubuntu:~/mavros/mavros/scripts$ sudo ./install_geographiclib_datasets.sh
    Installing GeographicLib geoids egm96-5
    Installing GeographicLib gravity egm96
    Installing GeographicLib magnetic emm2015

YOLOv8
   ![2500fdec-bd9c-426b-8bab-cff6bfec2acc](file:///D:/Typedown/2500fdec-bd9c-426b-8bab-cff6bfec2acc.png)

    s



## ### 图像增强算法

### 前提条件

| 构建类型 | Linux | MacOS | Windows |
| ---- | ----- | ----- | ------- |
| 脚本环境 | 已测试   | 待定    | 待定      |

此外，代码可在满足以下最低要求的情况下运行：

* 在Ubuntu 20.04 LTS系统上测试通过的依赖项如下：
  * Python 3.5.2
  * PyTorch '1.0.1.post2'
  * torchvision 0.2.2
  * opencv 4.0.0
  * scipy 1.2.1
  * numpy 1.16.2
  * tqdm

在Linux环境中安装，请执行：
    pip install -r requirements.txt



## 

|     |     |     |     |
| --- | --- | --- | --- |
|     |     |     |     |


