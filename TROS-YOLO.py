#!/usr/bin/env python3
# yolo_ros2_bpu.py
# 基于RDK X5 BPU加速的YOLO检测节点
# 支持模型量化、BPU硬件加速、多线程推理

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import cv2
import numpy as np
import time
import os
import json

# 地平线BPU推理库
from hobot_dnn import pyeasy_dnn as dnn
from hobot_vio import libsrcampy as srcampy

# ROS 2消息
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Header

class BPUModelLoader:
    """BPU模型加载与预处理工具类"""
    
    def __init__(self, model_path, class_names):
        """
        初始化BPU模型
        :param model_path: .bin模型文件路径
        :param class_names: 类别名称列表
        """
        self.classes = class_names
        self.model = None
        self.input_tensor = None
        self.output_tensors = []
        self.model_name = os.path.basename(model_path)
        self._load_model(model_path)
        
    def _load_model(self, model_path):
        """加载BPU模型并分析输入输出张量"""
        try:
            # 加载模型
            self.model = dnn.load(model_path)
            
            # 获取输入张量信息
            self.input_tensor = self.model[0].inputs[0]
            print(f"模型加载成功: {self.model_name}")
            print(f"输入张量: shape={self.input_tensor.properties.shape}, "
                  f"layout={self.input_tensor.properties.layout}")
            
            # 获取输出张量信息
            for i, output in enumerate(self.model[0].outputs):
                self.output_tensors.append(output)
                print(f"输出张量{i}: shape={output.properties.shape}, "
                      f"type={output.properties.dtype}, "
                      f"layout={output.properties.layout}")
                
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            raise RuntimeError("BPU模型加载错误")
    
    def get_input_shape(self):
        """获取模型输入尺寸 (height, width)"""
        if self.input_tensor is None:
            return (640, 640)
        shape = self.input_tensor.properties.shape
        # 地平线模型通常是NCHW布局
        if self.input_tensor.properties.layout == "NCHW":
            return (shape[2], shape[3])  # H, W
        return (shape[1], shape[2])  # 默认为NHWC
    
    def preprocess(self, image):
        """
        图像预处理
        :param image: 输入BGR图像
        :return: 预处理后的图像张量, 原始尺寸, 缩放比例, 填充信息
        """
        # 获取模型输入尺寸
        input_h, input_w = self.get_input_shape()
        
        # 原始尺寸
        orig_h, orig_w = image.shape[:2]
        
        # 保持宽高比的缩放
        scale = min(input_h / orig_h, input_w / orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        
        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h))
        
        # 创建填充图像
        padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # 转换为模型需要的布局 (NCHW)
        if self.input_tensor.properties.layout == "NCHW":
            # BGR to RGB
            padded = padded[:, :, ::-1]
            # HWC to CHW
            tensor_img = padded.transpose(2, 0, 1)
            # 添加batch维度
            tensor_img = np.expand_dims(tensor_img, axis=0)
        else:
            # 默认为NHWC
            tensor_img = np.expand_dims(padded, axis=0)
            
        return tensor_img, (orig_w, orig_h), scale, (0, 0, input_w - new_w, input_h - new_h)

class BPUYOLODetector:
    """YOLO检测器封装类，使用BPU加速"""
    
    def __init__(self, model_path, class_names, conf_thresh=0.5, iou_thresh=0.5):
        """
        初始化检测器
        :param model_path: .bin模型文件路径
        :param class_names: 类别名称列表
        :param conf_thresh: 置信度阈值
        :param iou_thresh: IOU阈值
        """
        self.model_loader = BPUModelLoader(model_path, class_names)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.color_map = self._generate_color_map(len(class_names))
        
        # 预热模型
        self._warm_up()
        
    def _warm_up(self):
        """预热模型，避免首次推理延迟"""
        dummy_input = np.zeros(self.model_loader.input_tensor.properties.shape, 
                              dtype=np.uint8)
        print("预热模型...")
        start = time.time()
        outputs = self.model_loader.model[0].forward(dummy_input)
        print(f"预热完成，耗时: {(time.time()-start)*1000:.2f}ms")
        
    def _generate_color_map(self, num_classes):
        """生成类别颜色映射"""
        np.random.seed(42)
        return [tuple(map(int, np.random.randint(0, 255, 3))) 
                for _ in range(num_classes)]
    
    def _postprocess(self, outputs, orig_size, scale, padding):
        """
        后处理：解析BPU输出为检测结果
        :param outputs: BPU推理输出
        :param orig_size: 原始图像尺寸 (w, h)
        :param scale: 缩放比例
        :param padding: 填充信息 (left, top, right, bottom)
        :return: 检测结果列表 [x1, y1, x2, y2, conf, cls_id]
        """
        detections = []
        orig_w, orig_h = orig_size
        pad_left, pad_top, pad_right, pad_bottom = padding
        
        # 地平线YOLO模型通常有三个输出层
        # 假设输出格式为: [batch, num_anchors, 5+num_classes]
        for i, output_tensor in enumerate(outputs):
            # 获取输出数据
            data = output_tensor.buffer
            
            # 解析输出维度
            if output_tensor.properties.layout == "NCHW":
                _, num_anchors, grid_h, grid_w = data.shape
                data = data.reshape(num_anchors, -1, grid_h, grid_w)
                data = data.transpose(0, 2, 3, 1)  # to NHWC
            else:
                _, grid_h, grid_w, num_anchors = data.shape
                data = data.reshape(-1, grid_h, grid_w, num_anchors)
            
            # 解析每个网格的检测结果
            for h in range(grid_h):
                for w in range(grid_w):
                    for a in range(num_anchors):
                        # 获取预测值
                        pred = data[0, h, w, a, :]
                        
                        # 提取边界框信息
                        tx, ty, tw, th, conf = pred[:5]
                        
                        # 应用置信度阈值
                        if conf < self.conf_thresh:
                            continue
                        
                        # 提取类别概率
                        cls_probs = pred[5:]
                        cls_id = np.argmax(cls_probs)
                        cls_conf = cls_probs[cls_id] * conf
                        
                        # 应用类别置信度阈值
                        if cls_conf < self.conf_thresh:
                            continue
                        
                        # 计算边界框坐标 (相对于特征图)
                        cx = (w + tx) / grid_w
                        cy = (h + ty) / grid_h
                        bw = tw / grid_w
                        bh = th / grid_h
                        
                        # 转换为像素坐标
                        x1 = int((cx - bw/2) * self.model_loader.get_input_shape()[1])
                        y1 = int((cy - bh/2) * self.model_loader.get_input_shape()[0])
                        x2 = int((cx + bw/2) * self.model_loader.get_input_shape()[1])
                        y2 = int((cy + bh/2) * self.model_loader.get_input_shape()[0])
                        
                        # 调整填充和缩放
                        x1 = max(0, x1 - pad_left)
                        y1 = max(0, y1 - pad_top)
                        x2 = max(0, x2 - pad_left)
                        y2 = max(0, y2 - pad_top)
                        
                        # 缩放到原始图像尺寸
                        x1 = int(x1 / scale)
                        y1 = int(y1 / scale)
                        x2 = int(x2 / scale)
                        y2 = int(y2 / scale)
                        
                        # 确保在图像范围内
                        x1 = max(0, min(orig_w, x1))
                        y1 = max(0, min(orig_h, y1))
                        x2 = max(0, min(orig_w, x2))
                        y2 = max(0, min(orig_h, y2))
                        
                        # 添加到检测结果
                        detections.append([x1, y1, x2, y2, cls_conf, cls_id])
        
        # 应用NMS
        return self._nms(detections)
    
    def _nms(self, detections):
        """非极大值抑制"""
        if len(detections) == 0:
            return []
        
        # 转换为numpy数组
        boxes = np.array([d[:4] for d in detections])
        scores = np.array([d[4] for d in detections])
        classes = np.array([d[5] for d in detections])
        
        # 按类别分组进行NMS
        final_detections = []
        for cls_id in np.unique(classes):
            mask = classes == cls_id
            cls_boxes = boxes[mask]
            cls_scores = scores[mask]
            
            # 按分数排序
            indices = np.argsort(-cls_scores)
            keep = []
            
            while indices.size > 0:
                # 取最高分
                i = indices[0]
                keep.append(i)
                
                # 计算IOU
                ious = self._bbox_iou(cls_boxes[i], cls_boxes[indices[1:]])
                
                # 保留IOU小于阈值的
                remaining = np.where(ious < self.iou_thresh)[0]
                indices = indices[remaining + 1]
            
            # 添加到最终结果
            for i in keep:
                x1, y1, x2, y2 = cls_boxes[i]
                final_detections.append([
                    x1, y1, x2, y2, cls_scores[i], cls_id
                ])
        
        return final_detections
    
    def _bbox_iou(self, box1, boxes):
        """计算单个框与多个框的IOU"""
        # 计算交集
        x1 = np.maximum(box1[0], boxes[:, 0])
        y1 = np.maximum(box1[1], boxes[:, 1])
        x2 = np.minimum(box1[2], boxes[:, 2])
        y2 = np.minimum(box1[3], boxes[:, 3])
        
        # 计算交集面积
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # 计算并集面积
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area1 + area2 - intersection
        
        # 计算IOU
        return intersection / union
    
    def detect(self, image):
        """
        执行目标检测
        :param image: 输入BGR图像
        :return: 检测结果列表 [x1, y1, x2, y2, conf, cls_id]
        """
        # 预处理
        start_pre = time.time()
        input_tensor, orig_size, scale, padding = self.model_loader.preprocess(image)
        pre_time = time.time() - start_pre
        
        # BPU推理
        start_inf = time.time()
        outputs = self.model_loader.model[0].forward(input_tensor)
        inf_time = time.time() - start_inf
        
        # 后处理
        start_post = time.time()
        detections = self._postprocess(outputs, orig_size, scale, padding)
        post_time = time.time() - start_post
        
        # 打印性能信息
        print(f"预处理: {pre_time*1000:.2f}ms, "
              f"推理: {inf_time*1000:.2f}ms, "
              f"后处理: {post_time*1000:.2f}ms, "
              f"总耗时: {(pre_time+inf_time+post_time)*1000:.2f}ms")
        
        return detections
    
    def visualize(self, image, detections):
        """
        可视化检测结果
        :param image: 原始图像
        :param detections: 检测结果
        :return: 绘制后的图像
        """
        img_draw = image.copy()
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            label = f"{self.model_loader.classes[int(cls_id)]} {conf:.2f}"
            
            # 绘制边界框
            color = self.color_map[int(cls_id)]
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签背景
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(img_draw, (x1, y1 - text_size[1] - 5), 
                         (x1 + text_size[0], y1), color, -1)
            
            # 绘制标签文本
            cv2.putText(img_draw, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img_draw

class YOLOCEHNode(Node):
    """ROS 2节点，使用BPU加速YOLO检测"""
    
    def __init__(self):
        super().__init__('yolo_ceh_detector_bpu')
        
        # 参数配置
        self.declare_parameter('model_path', '/opt/models/yolo_ceh.bin')
        self.declare_parameter('class_names', ['corrosion', 'crack', 'bio_fouling'])
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.5)
        self.declare_parameter('visualization', True)
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('detection_topic', '/detections')
        self.declare_parameter('visualization_topic', '/detection_visualization')
        
        # 获取参数
        model_path = self.get_parameter('model_path').value
        class_names = self.get_parameter('class_names').value
        conf_thresh = self.get_parameter('conf_threshold').value
        iou_thresh = self.get_parameter('iou_threshold').value
        self.visualization_enabled = self.get_parameter('visualization').value
        camera_topic = self.get_parameter('camera_topic').value
        detection_topic = self.get_parameter('detection_topic').value
        vis_topic = self.get_parameter('visualization_topic').value
        
        # 初始化检测器
        self.get_logger().info(f"加载BPU模型: {model_path}")
        self.detector = BPUYOLODetector(
            model_path, 
            class_names,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh
        )
        
        # 初始化工具
        self.bridge = CvBridge()
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0.0
        
        # 创建回调组（允许多线程）
        self.callback_group = ReentrantCallbackGroup()
        
        # 创建订阅和发布
        self.sub = self.create_subscription(
            Image,
            camera_topic,
            self.detect_cb,
            10,
            callback_group=self.callback_group
        )
        
        self.det_pub = self.create_publisher(
            Detection2DArray,
            detection_topic,
            10,
            callback_group=self.callback_group
        )
        
        if self.visualization_enabled:
            self.vis_pub = self.create_publisher(
                Image,
                vis_topic,
                10,
                callback_group=self.callback_group
            )
        
        self.get_logger().info("YOLO CEH BPU节点已启动")
        self.get_logger().info(f"使用类别: {class_names}")
        self.get_logger().info(f"置信度阈值: {conf_thresh}, IOU阈值: {iou_thresh}")

    def detect_cb(self, msg):
        try:
            # 转换ROS图像消息到OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # 执行检测
            detections = self.detector.detect(cv_image)
            
            # 转换为ROS消息
            det_msg = self._detections_to_ros(msg.header, detections)
            
            # 发布检测结果
            self.det_pub.publish(det_msg)
            
            # 发布可视化结果
            if self.visualization_enabled:
                vis_img = self.detector.visualize(cv_image, detections)
                vis_msg = self.bridge.cv2_to_imgmsg(vis_img, 'bgr8')
                vis_msg.header = msg.header
                self.vis_pub.publish(vis_msg)
            
            # 计算并打印FPS
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                current_time = time.time()
                elapsed = current_time - self.last_time
                self.fps = 30 / elapsed
                self.get_logger().info(f"处理帧率: {self.fps:.2f}fps")
                self.last_time = current_time
                
        except Exception as e:
            self.get_logger().error(f"处理图像时出错: {str(e)}")

    def _detections_to_ros(self, header, detections):
        """将检测结果转换为ROS消息"""
        det_msg = Detection2DArray()
        det_msg.header = header
        
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            
            detection = Detection2D()
            detection.header = header
            
            # 边界框
            bbox = BoundingBox2D()
            center = Pose2D()
            center.x = (x1 + x2) / 2.0
            center.y = (y1 + y2) / 2.0
            bbox.center = center
            bbox.size_x = float(x2 - x1)
            bbox.size_y = float(y2 - y1)
            detection.bbox = bbox
            
            # 结果
            result = ObjectHypothesis()
            result.id = int(cls_id)
            result.score = float(conf)
            detection.results.append(result)
            
            det_msg.detections.append(detection)
        
        return det_msg

def main(args=None):
    rclpy.init(args=args)
    
    try:
        # 创建节点
        node = YOLOCEHNode()
        
        # 使用多线程执行器
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        
        # 打印启动信息
        node.get_logger().info("启动YOLO CEH BPU节点...")
        node.get_logger().info("使用多线程执行器，线程数: 4")
        
        try:
            executor.spin()
        finally:
            executor.shutdown()
            node.destroy_node()
            
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()