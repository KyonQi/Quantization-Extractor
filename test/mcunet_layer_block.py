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
b_data = parse_log('./test/coordinator_mcunet_block.log')
l_data = parse_log('./test/coordinator_mcunet_layer.log')
h_data = parse_log('./test/coordinator_mcunet_hybrid.log')

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

plt.savefig('./test/mcunet_three_mode_comparison.png')
