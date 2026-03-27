import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_log(filename):
    data = []
    pattern = r"Layer\s+([\d-]+)\s+\[\s*(\w+)\]\s+([\w\+]+):\s+total=([\d.]+)ms\s+compute=([\d.]+)ms"
    with open(filename, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                layer_range, _, name, total, compute = match.groups()
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


# 1. 载入数据
b_data = parse_log('./test/coordinator_mcunet_pw_block.log')
l_data = parse_log('./test/coordinator_mcunet_pw_layer.log')
h_data = parse_log('./test/coordinator_mcunet_pw_hybrid.log')

# 2. 按 Block 边界对齐
l_comp_map = {item['indices'][0]: item['compute'] for item in l_data}
l_tot_map = {item['indices'][0]: item['total'] for item in l_data}

comparison = []
for b in b_data:
    b_idx = set(b['indices'])

    l_sum_tot = sum(l_tot_map[i] for i in b_idx if i in l_tot_map)
    l_sum_comp = sum(l_comp_map[i] for i in b_idx if i in l_comp_map)

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

# 计算通信开销
df['B_Comm'] = df['B_Tot'] - df['B_Comp']
df['L_Comm'] = df['L_Tot'] - df['L_Comp']
df['H_Comm'] = df['H_Tot'] - df['H_Comp']

labels = [f"{n}\n({r})" for n, r in zip(df['Block'], df['Range'])]
x = np.arange(len(df))
width = 0.25

# 颜色方案
color_fill = {'L': '#e74c3c', 'B': '#3498db', 'H': '#2ecc71'}
color_edge = {'L': '#922b21', 'B': '#1a5276', 'H': '#196f3d'}
color_text = {'L': '#922b21', 'B': '#1a5276', 'H': '#196f3d'}

fig, ax = plt.subplots(figsize=(18, 8))

# 按 Layer, Block, Hybrid 顺序排列
ax.bar(x - width, df['L_Comm'], width, label='Layer',
       color=color_fill['L'], edgecolor=color_edge['L'], linewidth=0.8)
ax.bar(x, df['B_Comm'], width, label='Block',
       color=color_fill['B'], edgecolor=color_edge['B'], linewidth=0.8)
ax.bar(x + width, df['H_Comm'], width, label='Hybrid',
       color=color_fill['H'], edgecolor=color_edge['H'], linewidth=0.8)

# 标注相对于 Layer 的通信降幅（↓xx%），错开高度避免重叠
y_max = max(df['L_Comm'].max(), df['B_Comm'].max(), df['H_Comm'].max())
pad = y_max * 0.03

for i in range(len(df)):
    l_comm = df['L_Comm'].iloc[i]
    if l_comm > 0:
        bar_top = max(df['B_Comm'].iloc[i], df['H_Comm'].iloc[i])
        # Block 降幅 — 较低位置
        b_saving = (l_comm - df['B_Comm'].iloc[i]) / l_comm * 100
        if abs(b_saving) > 1:
            ax.text(x[i], bar_top + pad, f'B ↓{b_saving:.0f}%',
                    ha='center', va='bottom', fontsize=7.5, color=color_text['B'], fontweight='bold')
        # Hybrid 降幅 — 较高位置，错开避免重叠
        h_saving = (l_comm - df['H_Comm'].iloc[i]) / l_comm * 100
        if abs(h_saving) > 1:
            ax.text(x[i], bar_top + pad * 3.5, f'H ↓{h_saving:.0f}%',
                    ha='center', va='bottom', fontsize=7.5, color=color_text['H'], fontweight='bold')

# 汇总统计框
total_l_comm = df['L_Comm'].sum()
total_b_comm = df['B_Comm'].sum()
total_h_comm = df['H_Comm'].sum()
summary = (
    f"Total Communication Overhead:\n"
    f"  Layer:  {total_l_comm:>8.2f} ms  ({(0) / total_l_comm * 100:+.1f}%)\n"
    f"  Block:  {total_b_comm:>8.2f} ms  ({(total_b_comm - total_l_comm) / total_l_comm * 100:+.1f}%)\n"
    f"  Hybrid: {total_h_comm:>8.2f} ms  ({(total_h_comm - total_l_comm) / total_l_comm * 100:+.1f}%)"
)
ax.text(0.98, 0.95, summary, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fdfefe', alpha=0.95, edgecolor='#bdc3c7'),
        family='monospace')

ax.set_xlabel('Block Partition (Layer Range)', fontsize=11)
ax.set_ylabel('Communication Time (ms)', fontsize=11)
ax.set_title('Communication Overhead Comparison: Layer vs. Block vs. Hybrid',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('./test/mcunet_comm_analysis_2.png', dpi=300)
