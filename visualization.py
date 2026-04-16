from openpyxl import load_workbook
import json
import statistics
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.font_manager as fm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

bad_bids = []
#for i in range(len(bad_bids)):
#    bad_bids[i] += 1  # 转换为从1开始的索引

if __name__ == "__main__":
    # 加载原始Excel文件
    workbook = load_workbook(filename='.\\数据\\testdata_masked.xlsx')
    worksheet = workbook.active

    xm_list, zb_list, tb_list = [], [], []
    connecter = ["projguid", "bid_ann_guid"]

    bad_table = []
    for cell in worksheet['B']:
        if cell.value != None:
            bad_table.append(int(cell.value))
    if bad_table != []:
        bad_bids = []
        for i in range(len(bad_table)):
            if bad_table[i] == 1:
                bad_bids.append(i+1)  # 转换为0-based索引
    print(f"异常招标索引: {bad_bids}, 共{len(bad_bids)}个")

    # 读取A列数据并处理
    for cell in worksheet['A']:
        if cell.value[0] != "{":
            continue
        dict_str = cell.value
        if dict_str:
            try:
                # 解析字典字符串
                data_dict = json.loads(dict_str)
                # 根据connecter分类
                if connecter[0] in data_dict and connecter[1] not in data_dict:
                    xm_list.append(data_dict)
                elif connecter[1] in data_dict and connecter[0] not in data_dict:
                    tb_list.append(data_dict)
                elif connecter[0] in data_dict and connecter[1] in data_dict:
                    zb_list.append(data_dict)
                else:
                    assert(False)
            except json.JSONDecodeError as e:
                print(f"第{cell.row}行解析失败: {e}")
                continue
    print(f"项目数量: {len(xm_list)}, 招标数量: {len(zb_list)}, 投标数量: {len(tb_list)}")

    x_axis = []
    y_axis = []
    y2_axis = []  # 存储标准差与算术平均数的比值
    index_list = []  # 存储每个点的_index

    for zb in zb_list:
        tb_group = []
        for tb in tb_list:
            if zb['bid_ann_guid'] == tb['bid_ann_guid']:
                tb_group.append(tb)

        tb_group.sort(key=lambda x: x['x_price'])

        # 分离出中标者和落败者的出价
        winning_bids = [bid['x_price'] for bid in tb_group if bid['x_isqualified'] == 1]
        losing_bids = [bid['x_price'] for bid in tb_group if bid['x_isqualified'] == 0]

        # 计算两个最低出价之差Δ
        if len(tb_group) >= 2:
            lowest_bid = tb_group[0]['x_price']
            second_lowest_bid = tb_group[1]['x_price']
            delta = second_lowest_bid - lowest_bid
        else:
            delta = 0  # 出价数量不足时无法计算

        # 计算落败竞价的标准差σ
        if len(losing_bids) >= 2:
            sigma = statistics.stdev(losing_bids)
        elif len(losing_bids) == 1:
            sigma = 0  # 只有一个落败出价时标准差为0
        else:
            sigma = 0

        # 计算比值Δ/σ
        if sigma != 0:
            ratio = delta / sigma
        else:
            ratio = float('inf')  # 如果标准差为0，比值设为无穷大

        # 计算所有投标价格的标准差与算术平均数的比值
        all_bids = [bid['x_price'] for bid in tb_group]
        if len(all_bids) >= 2:
            mean_price = statistics.mean(all_bids)
            std_price = statistics.stdev(all_bids)
            if mean_price != 0:
                std_mean_ratio = std_price / mean_price
            else:
                std_mean_ratio = float('inf')
        else:
            std_mean_ratio = 0  # 出价数量不足时无法计算

        x_axis.append(zb['creationdate'])
        y_axis.append(ratio)
        y2_axis.append(std_mean_ratio)  # 新增：收集标准差与算术平均数的比值

    index_list = [i+1 for i in range(len(zb_list))]  # 使用顺序编号作为_index
    
    # 处理日期格式
    dates = [datetime.strptime(date, '%Y-%m-%d') for date in x_axis]
    
    # 过滤掉无穷大的值，以便更好地显示图表
    valid_dates = []
    valid_ratios = []
    valid_std_mean_ratios = []  # 过滤后的标准差与算术平均数比值
    valid_indices = []  # 新增：存储有效点的_index
    for date, ratio, std_mean_ratio, idx in zip(dates, y_axis, y2_axis, index_list):
        if ratio != float('inf') and ratio <= 10:
            valid_dates.append(date)
            valid_ratios.append(ratio)
            valid_std_mean_ratios.append(std_mean_ratio)
            valid_indices.append(idx)
    
    # 将数据分为两组：bad_bids中的点和正常点
    bad_dates = []
    bad_ratios = []
    bad_std_mean_ratios = []
    bad_indices = []
    normal_dates = []
    normal_ratios = []
    normal_std_mean_ratios = []
    normal_indices = []
    
    for date, ratio, std_mean_ratio, idx in zip(valid_dates, valid_ratios, valid_std_mean_ratios, valid_indices):
        if idx in bad_bids:
            bad_dates.append(date)
            bad_ratios.append(ratio)
            bad_std_mean_ratios.append(std_mean_ratio)
            bad_indices.append(idx)
        else:
            normal_dates.append(date)
            normal_ratios.append(ratio)
            normal_std_mean_ratios.append(std_mean_ratio)
            normal_indices.append(idx)
    
    # 计算第一张图（掩护出价显示）的平均值
    if normal_ratios:
        normal_mean_1 = statistics.mean(normal_ratios)
    else:
        normal_mean_1 = 0
        
    if bad_ratios:
        bad_mean_1 = statistics.mean(bad_ratios)
    else:
        bad_mean_1 = 0
        
    # 计算第二张图（投标价格变异系数显示）的平均值
    if normal_std_mean_ratios:
        normal_mean_2 = statistics.mean(normal_std_mean_ratios)
    else:
        normal_mean_2 = 0
        
    if bad_std_mean_ratios:
        bad_mean_2 = statistics.mean(bad_std_mean_ratios)
    else:
        bad_mean_2 = 0
    
    point_size = 20  # 点的大小
    
    # 创建两个并列的图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    
    # 第一张图：掩护出价显示
    # 绘制正常点（蓝色）
    ax1.scatter(normal_dates, normal_ratios, alpha=0.6, color='b', s=point_size, label='正常项目')
    
    # 绘制bad_bids中的点（红色）
    ax1.scatter(bad_dates, bad_ratios, alpha=0.8, color='r', s=point_size, label='异常项目')
    
    # 为所有点添加标签（_index）
    for i, idx in enumerate(normal_indices):
        ax1.annotate(
            str(idx),  # 标签文本
            (normal_dates[i], normal_ratios[i]),  # 标签位置（点的坐标）
            textcoords="offset points",  # 文本相对于点的偏移
            xytext=(5, 5),  # 标签相对于点的偏移量
            ha='center',  # 水平对齐方式
            fontsize=8,  # 标签字体大小
            color='black',  # 标签颜色
            alpha=0.7  # 标签透明度
        )
    
    for i, idx in enumerate(bad_indices):
        ax1.annotate(
            str(idx),  # 标签文本
            (bad_dates[i], bad_ratios[i]),  # 标签位置（点的坐标）
            textcoords="offset points",  # 文本相对于点的偏移
            xytext=(5, 5),  # 标签相对于点的偏移量
            ha='center',  # 水平对齐方式
            fontsize=8,  # 标签字体大小
            color='black',  # 红色点使用黑色标签以便更好地显示
            alpha=0.9,  # 标签透明度
            weight='bold'  # 加粗标签
        )
    
    # 设置日期格式器
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 每3个月显示一个日期标签
    
    # 设置坐标轴标签和标题
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('相对距离值 (最低两个出价差值/未中标出价标准差)', fontsize=12)
    ax1.set_title('相对距离散点图', fontsize=14)
    
    # 添加图例
    ax1.legend()
    
    # 设置y轴范围
    ax1.set_ylim(0, 7)
    
    # 旋转日期标签以便更好地显示
    plt.setp(ax1.get_xticklabels(), rotation=45)
    
    # 添加网格线
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.axhline(y=1, color='y', linestyle='--', linewidth=1.5, alpha=0.8)
    
    # 添加平均值线
    ax1.axhline(y=normal_mean_1, color='b', linestyle='-', linewidth=2, alpha=0.7, label=f'正常项目平均值: {normal_mean_1:.2f}')
    ax1.axhline(y=bad_mean_1, color='r', linestyle='-', linewidth=2, alpha=0.7, label=f'异常项目平均值: {bad_mean_1:.2f}')
    ax1.legend()
    
    # 第二张图：标准差与算术平均数的比值
    # 绘制正常点（蓝色）
    ax2.scatter(normal_dates, normal_std_mean_ratios, alpha=0.6, color='b', s=point_size, label='正常项目')
    
    # 绘制bad_bids中的点（红色）
    ax2.scatter(bad_dates, bad_std_mean_ratios, alpha=0.8, color='r', s=point_size, label='异常项目')
    
    # 为所有点添加标签（_index）
    for i, idx in enumerate(normal_indices):
        ax2.annotate(
            str(idx),  # 标签文本
            (normal_dates[i], normal_std_mean_ratios[i]),  # 标签位置（点的坐标）
            textcoords="offset points",  # 文本相对于点的偏移
            xytext=(5, 5),  # 标签相对于点的偏移量
            ha='center',  # 水平对齐方式
            fontsize=8,  # 标签字体大小
            color='black',  # 标签颜色
            alpha=0.7  # 标签透明度
        )
    
    for i, idx in enumerate(bad_indices):
        ax2.annotate(
            str(idx),  # 标签文本
            (bad_dates[i], bad_std_mean_ratios[i]),  # 标签位置（点的坐标）
            textcoords="offset points",  # 文本相对于点的偏移
            xytext=(5, 5),  # 标签相对于点的偏移量
            ha='center',  # 水平对齐方式
            fontsize=8,  # 标签字体大小
            color='black',  # 红色点使用黑色标签以便更好地显示
            alpha=0.9,  # 标签透明度
            weight='bold'  # 加粗标签
        )
    
    # 设置日期格式器
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 每3个月显示一个日期标签
    
    # 设置坐标轴标签和标题
    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_ylabel('变异系数 (标准差/算术平均数)', fontsize=12)
    ax2.set_title('投标价格变异系数显示', fontsize=14)
    
    # 添加图例
    ax2.legend()
    
    # 设置y轴范围（根据数据自动调整）
    ax2.set_ylim(0, max(valid_std_mean_ratios) * 1.1 if valid_std_mean_ratios else 1)
    
    # 旋转日期标签以便更好地显示
    plt.setp(ax2.get_xticklabels(), rotation=45)
    
    # 添加网格线
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 添加平均值线
    ax2.axhline(y=normal_mean_2, color='b', linestyle='-', linewidth=2, alpha=0.7, label=f'正常项目平均值: {normal_mean_2:.4f}')
    ax2.axhline(y=bad_mean_2, color='r', linestyle='-', linewidth=2, alpha=0.7, label=f'异常项目平均值: {bad_mean_2:.4f}')
    ax2.legend()
    
    # 调整布局
    plt.tight_layout()

    # 显示图表
    plt.show()