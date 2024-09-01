import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ['Micro-F1', 'NMI']  # 调换位置
acm_values = {
    'w/o NLE': [95.72, 85.75],
    'w/o GLD': [95.82, 82.12],
    'w/o RCL': [95.52, 87.57],
    'MLHGAE': [97.15, 92.27]
}

dblp_values = {
    'w/o NLE': [96.02, 91.05],
    'w/o GLD': [93.12, 80.12],
    'w/o RCL': [96.02, 89.71],
    'MLHGAE': [96.93, 91.33]
}

bar_width = 0.2
index = np.arange(len(categories))

# 颜色
colors = ['#00A4EF', '#F25022', '#FFB900', '#7FBA00']

# 创建主图，用于合并两个子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9), sharey=False)

# 绘制第一个柱状图
handles = []
labels = []
for i, (label, values) in enumerate(acm_values.items()):
    bar = ax1.bar(index + i * bar_width, values, bar_width, color=colors[i], label=label)
    handles.append(bar)
    labels.append(label)
ax1.set_xlabel('(a) ACM', fontsize=28, fontname='Times New Roman')
ax1.set_xticks(index + 1.5 * bar_width)
ax1.set_xticklabels(categories, fontsize=28, fontname='Times New Roman')
ax1.set_ylim(75, 100)
ax1.tick_params(axis='y', labelsize=20)

# 绘制第二个柱状图
for i, (label, values) in enumerate(dblp_values.items()):
    ax2.bar(index + i * bar_width, values, bar_width, color=colors[i], label=label)
ax2.set_xlabel('(b) DBLP', fontsize=28, fontname='Times New Roman')
ax2.set_xticks(index + 1.5 * bar_width)
ax2.set_xticklabels(categories, fontsize=28, fontname='Times New Roman')
ax2.set_ylim(75, 100)
ax2.tick_params(axis='y', labelsize=20)

# 调整图例位置并添加图例
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2,
           fontsize=22, prop={'family': 'Times New Roman', 'size': 28})

# # 通过增加顶部边距来为图例腾出空间
# plt.subplots_adjust(top=0.85)

# 调整布局，确保图例显示在图片中
plt.tight_layout(rect=[0, 0, 1, 0.85])
plt.show()
