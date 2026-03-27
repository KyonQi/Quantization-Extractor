# import re
# import matplotlib.pyplot as plt
# import pandas as pd

# def parse_inference_log(filename):
#     """
#     解析日志文本，提取各层的执行指标。
#     """
#     records = []
#     # 正则表达式捕获层范围、名称、总耗时(Total)与计算耗时(Compute)
#     pattern = r"Layer\s+([\d-]+)\s+\[\s*\w+\]\s+([\w\+]+):\s+total=([\d.]+)ms\s+compute=([\d.]+)ms"
    
#     with open(filename, 'r') as f:
#         for line in f:
#             match = re.search(pattern, line)
#             if match:
#                 l_range, name, total, compute = match.groups()
#                 # 处理层索引映射逻辑
#                 if '-' in l_range:
#                     start, end = map(int, l_range.split('-'))
#                     indices = list(range(start, end + 1))
#                 else:
#                     indices = [int(l_range)]
                    
#                 records.append({
#                     'indices': indices,
#                     'total': float(total),
#                     'compute': float(compute),
#                     'name': name,
#                     'range': l_range
#                 })
#     return records

# # 1. 载入实验原始数据
# block_log_data = parse_inference_log('./test/coordinator_035_block_padd.log')
# layer_log_data = parse_inference_log('./test/coordinator_035_layer_padd.log')

# # 2. 构建层索引查询表 (Layer Mode)
# layer_map = {item['indices'][0]: item for item in layer_log_data}

# # 3. 对齐数据：将 Layer 模式的离散耗时按 Block 边界进行累加
# comparison_data = []
# for b_entry in block_log_data:
#     target_indices = b_entry['indices']
    
#     # 算术求和：对应索引在 Layer 模式下的总耗时
#     sum_total_layer = sum(layer_map[idx]['total'] for idx in target_indices if idx in layer_map)
#     sum_compute_layer = sum(layer_map[idx]['compute'] for idx in target_indices if idx in layer_map)
    
#     comparison_data.append({
#         'Block_Name': b_entry['name'],
#         'Layer_Range': b_entry['range'],
#         'Block_Mode_Total_ms': b_entry['total'],
#         'Layer_Mode_Accumulated_ms': sum_total_layer,
#         'Absolute_Saving_ms': sum_total_layer - b_entry['total'],
#         'Efficiency_Gain_Pct': ((sum_total_layer - b_entry['total']) / sum_total_layer * 100) 
#                                 if sum_total_layer > 0 else 0
#     })

# # 4. 导出结构化分析结果
# df_comparison = pd.DataFrame(comparison_data)
# # df_comparison.to_csv('mbv2_035_full_analysis.csv', index=False)

# # 5. 生成定量对比图表
# plt.figure(figsize=(15, 8))
# x_axis = range(len(df_comparison))
# bar_width = 0.35

# plt.bar([i - bar_width/2 for i in x_axis], df_comparison['Block_Mode_Total_ms'], 
#         bar_width, label='Block Mode Total', color='#2C3E50')
# plt.bar([i + bar_width/2 for i in x_axis], df_comparison['Layer_Mode_Accumulated_ms'], 
#         bar_width, label='Layer Mode (Accumulated)', color='#95A5A6')

# plt.xlabel('Logical Block Partition (Layer Index)')
# plt.ylabel('Latency (ms)')
# plt.title('MobileNetV2 0.35 Performance Benchmarking: Block vs. Layer Scheduling')
# plt.xticks(x_axis, [f"{n}\n({r})" for n, r in zip(df_comparison['Block_Name'], df_comparison['Layer_Range'])], 
#            rotation=45, ha='right', fontsize=8)
# plt.legend()
# plt.grid(axis='y', linestyle='--', alpha=0.4)
# plt.tight_layout()

# plt.savefig('./test/mbv2_035_comparison_plot.png')

import re
import matplotlib.pyplot as plt
import pandas as pd

def parse_log(filename):
    data = []
    # 提取：层索引、块名称、总耗时(total)、计算耗时(compute)
    pattern = r"Layer\s+([\d-]+)\s+\[\s*(\w+)\]\s+([\w\+]+):\s+total=([\d.]+)ms\s+compute=([\d.]+)ms"
    with open(filename, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                layer_range, _, name, total, compute = match.groups()
                # 处理层索引范围
                idx_parts = [int(x) for x in layer_range.split('-')]
                indices = list(range(idx_parts[0], idx_parts[1] + 1)) if '-' in layer_range else [int(layer_range)]
                
                data.append({
                    'indices': indices, 
                    'total': float(total), 
                    'compute': float(compute), 
                    'name': name, 
                    'range': layer_range
                })
    return data

# 1. 载入原始数据
b_data = parse_log('./test/coordinator_035_pw_block.log')
l_data = parse_log('./test/coordinator_035_pw_layer.log')
h_data = parse_log('./test/coordinator_035_pw_hybrid.log')

# 2. 构建 Layer 模式的映射 (按 Block 边界对齐)
# 这里的逻辑是：将 Block 模式中对应的多个层的计算耗时和总耗时分别累加
l_comp_map = {item['indices'][0]: item['compute'] for item in l_data}
l_tot_map = {item['indices'][0]: item['total'] for item in l_data}

comparison = []
for b in b_data:
    b_idx = set(b['indices'])
    
    # Layer Mode 累加
    l_sum_tot = sum(l_tot_map[i] for i in b_idx if i in l_tot_map)
    l_sum_comp = sum(l_comp_map[i] for i in b_idx if i in l_comp_map)
    
    # Hybrid Mode 累加 (查找包含在当前 Block 索引范围内的所有 Hybrid 记录)
    h_subsets = [h for h in h_data if set(h['indices']).issubset(b_idx)]
    h_sum_tot = sum(h['total'] for h in h_subsets)
    h_sum_comp = sum(h['compute'] for h in h_subsets)
    
    comparison.append({
        'Block': b['name'], 'Range': b['range'],
        'B_Tot': b['total'], 'B_Comp': b['compute'],
        'L_Tot': l_sum_tot, 'L_Comp': l_sum_comp,
        'H_Tot': h_sum_tot, 'H_Comp': h_sum_comp
    })

df = pd.DataFrame(comparison)

# 3. 绘图：三组堆叠柱状图
plt.figure(figsize=(18, 10))
x = range(len(df))
width = 0.25  # 每组模式的宽度

# 颜色配置：深色代表 Compute，浅色（或带透明度）代表 Communicate
colors = {
    'B': ('#2980b9', '#3498db'), # Block: 深蓝/浅蓝
    'L': ('#c0392b', '#e74c3c'), # Layer: 深红/浅红
    'H': ('#27ae60', '#2ecc71')  # Hybrid: 深绿/浅绿
}

# --- Block Mode ---
plt.bar([i - width for i in x], df['B_Comp'], width, label='Block: Compute', color=colors['B'][0])
plt.bar([i - width for i in x], df['B_Tot'] - df['B_Comp'], width, bottom=df['B_Comp'], 
        label='Block: Communicate', color=colors['B'][1], alpha=0.5)

# --- Layer Mode ---
plt.bar(x, df['L_Comp'], width, label='Layer: Compute', color=colors['L'][0])
plt.bar(x, df['L_Tot'] - df['L_Comp'], width, bottom=df['L_Comp'], 
        label='Layer: Communicate', color=colors['L'][1], alpha=0.5)

# --- Hybrid Mode ---
plt.bar([i + width for i in x], df['H_Comp'], width, label='Hybrid: Compute', color=colors['H'][0])
plt.bar([i + width for i in x], df['H_Tot'] - df['H_Comp'], width, bottom=df['H_Comp'], 
        label='Hybrid: Communicate', color=colors['H'][1], alpha=0.5)

b_total = df['B_Tot'].sum() / 1000  # 转换为秒
l_total = df['L_Tot'].sum() / 1000
h_total = df['H_Tot'].sum() / 1000
summary_box = (
    f"Total Inference Time:\n"
    f"Block Mode: {b_total:.4f} s\n"
    f"Layer Mode: {l_total:.4f} s\n"
    f"Hybrid Mode: {h_total:.4f} s"
)
plt.text(0.98, 0.95, summary_box, transform=plt.gca().transAxes, fontsize=12, 
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

# 4. 细节调整
plt.xticks(x, [f"{n}\n({r})" for n, r in zip(df['Block'], df['Range'])], rotation=45, ha='right', fontsize=9)
plt.ylabel('Latency (ms)')
plt.title('Detailed Performance Decomposition: Compute vs. Communicate')
plt.legend(ncol=3, loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()

plt.savefig('./test/mbv2_035_three_mode_comparison.png')