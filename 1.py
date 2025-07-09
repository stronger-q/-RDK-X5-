import matplotlib.pyplot as plt
import numpy as np

# 原始数据
optimized_before = {
    "延迟范围": ["0-2", "2-5", "5-10", ">10"],
    "频率": [5, 30, 60, 5],
    "类型": "优化前"
}

optimized_after = {
    "延迟范围": ["0-1", "1-1.5", ">1.5"],
    "频率": [85, 12, 3],
    "类型": "优化后"
}

# 创建画布
plt.figure(figsize=(10, 6), dpi=100)
plt.style.use('seaborn-whitegrid')

# 转换数据点为连续坐标
def process_data(data):
    x_labels = []
    x_values = []
    for r in data["延迟范围"]:
        if '-' in r:
            start, end = map(float, r.split('-'))
            x_labels.append(r + " ms")
            x_values.append((start + end)/2)
        else:
            x_labels.append(r + " ms")
            x_values.append(float(r[1:]) + 0.25)
    return x_values, data["频率"], x_labels

# 绘制折线
for dataset in [optimized_before, optimized_after]:
    x, y, labels = process_data(dataset)
    plt.plot(x, y,
             marker='o',
             linewidth=2,
             markersize=8,
             label=dataset["类型"])

# 坐标轴设置
plt.xticks(ticks=np.arange(0, 11, 0.5),
           labels=[f"{x:.1f}" if x%1!=0 else f"{int(x)}" for x in np.arange(0, 11, 0.5)])
plt.yticks(range(0, 101, 10))

plt.xlabel("延迟时间 (ms)", fontsize=12, labelpad=15)
plt.ylabel("频率 (%)", fontsize=12, labelpad=15)
plt.title("OpenAMP优化前后延迟分布对比", fontsize=14, pad=20)

# 添加特殊标注
plt.annotate('优化后85%请求\n<1ms响应',
             xy=(0.5, 85), xytext=(1.5, 75),
             arrowprops=dict(arrowstyle="->"),
             fontsize=10)

plt.annotate('优化前主要延迟\n集中在5-10ms',
             xy=(7.5, 60), xytext=(5, 70),
             arrowprops=dict(arrowstyle="->"),
             fontsize=10)

# 辅助元素
plt.grid(True, alpha=0.4, linestyle='--')
plt.legend(fontsize=12)
plt.tight_layout()

# 显示图形
plt.show()