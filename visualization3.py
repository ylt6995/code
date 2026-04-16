import matplotlib.pyplot as plt
from openpyxl import load_workbook
import statistics

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 文件路径
#result_path = ['.\实验结果\output_method1_time.xlsx', '.\实验结果\output_method2_time.xlsx', '.\实验结果\output_method3_time.xlsx']
result_path = ['.\\实验结果\\output_method1_time.xlsx', '.\\实验结果\\output_method1_gpt.xlsx']

method_names = ['glm-4-flash', 'gpt-4o-mini']
colors = ['blue', 'orange']

# 读取数据
avg_times = []
min_times = []
max_times = []
std_devs = []  # 新增：存储标准差

for file_path in result_path:
    try:
        # 加载Excel文件
        workbook = load_workbook(filename=file_path)
        worksheet = workbook.active
        
        # 读取E列数据（运行时间）
        times = []
        for cell in worksheet['E']:
            if cell.value and isinstance(cell.value, (int, float)):
                times.append(cell.value)
        
        if times:
            # 计算平均值、最小值和最大值
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            std_dev = statistics.stdev(times)
            
            avg_times.append(avg_time)
            min_times.append(min_time)
            max_times.append(max_time)
            std_devs.append(std_dev)  # 新增：存储标准差
        else:
            print(f"文件 {file_path} 中没有有效的时间数据")
            avg_times.append(0)
            min_times.append(0)
            max_times.append(0)
            std_devs.append(0)  # 新增：存储标准差
            
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        avg_times.append(0)
        min_times.append(0)
        max_times.append(0)
        std_devs.append(0)  # 新增：存储标准差

# 计算误差值（用于误差棒）
error_lower = [avg - min_val for avg, min_val in zip(avg_times, min_times)]
error_upper = [max_val - avg for avg, max_val in zip(avg_times, max_times)]
error = [error_lower, error_upper]

# 创建柱状图
plt.figure(figsize=(8, 7))

# 绘制柱状图，添加误差棒，调整柱子宽度为0.5
bar_width = 0.35
bars = plt.bar(method_names, avg_times, width=bar_width, color=colors, yerr=error, capsize=5, alpha=0.7)

# 范围标签移到柱子侧面
for bar, avg_time, min_time, max_time, std_dev in zip(bars, avg_times, min_times, max_times, std_devs):
    plt.text(bar.get_x() + bar.get_width()/2. - 0.25, avg_time + error_upper[bars.index(bar)] + 2, 
             f'范围: [{min_time:.2f}, {max_time:.2f}]', 
             ha='left', va='center', fontsize=9)
    plt.text(bar.get_x() + bar.get_width()/2. - 0.25, avg_time + error_upper[bars.index(bar)] + 4, 
             f'标准差: {std_dev:.2f}', 
             ha='left', va='center', fontsize=9)

# 添加平均值虚线和y轴标注
for i, avg_time in enumerate(avg_times):
    if avg_time > 0:  # 只处理有效数据
        # 绘制水平虚线，从y轴到柱子
        plt.hlines(y=avg_time, xmin=-0.6, xmax=bars[i].get_x() + bar_width/2, 
                  linestyle='--', color=colors[i], alpha=0.8)
        
        # 在y轴上标注平均值
        plt.text(-0.6, avg_time, f'{avg_time:.2f}', 
                ha='right', va='center', fontsize=10, 
                color=colors[i], weight='bold')

# 设置图表标题和坐标轴标签
plt.title('两种模型运行时间对比', fontsize=14, pad=20)
#plt.xlabel('方法', fontsize=12, labelpad=10)
plt.ylabel('运行时间 (秒)', fontsize=12, labelpad=10)

# 添加网格线
#plt.grid(True, linestyle='--', alpha=0.7, axis='y')

# 调整y轴范围，为顶部留出空间
max_y = max(avg_times) + max(error_upper) + 1
plt.ylim(0, max_y)

# 调整x轴范围，为左侧标注留出空间
plt.xlim(-0.6, len(method_names) - 0.4)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()