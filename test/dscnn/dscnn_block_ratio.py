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


def build_df(b_data):
    """直接从 Block 模式数据构建 DataFrame"""
    rows = []
    for item in b_data:
        comm = item['total'] - item['compute']
        rows.append({
            'name': item['name'],
            'range': item['range'],
            'total': item['total'],
            'compute': item['compute'],
            'comm': comm,
            'comm_pct': comm / item['total'] * 100 if item['total'] > 0 else 0,
        })
    return pd.DataFrame(rows)


# =====================================================================
# 载入数据
# =====================================================================
dfs = {}
for n_mcu, folder in [(3, '3_mcus'), (4, '4_mcus'), (5, '5_mcus'), (6, '6_mcus')]:
    b = parse_log(f'./test/{folder}/coordinator_dscnn_pw_block.log')
    dfs[n_mcu] = build_df(b)

df3 = dfs[3]
df4 = dfs[4]
df5 = dfs[5]
df6 = dfs[6]
labels = [f"{n}\n({r})" for n, r in zip(df4['name'], df4['range'])]
x = np.arange(len(df4))

# 颜色方案
color_fill = {'3': '#f39c12', '4': '#e74c3c', '5': '#3498db', '6': '#9b59b6'}
color_edge = {'3': '#d35400', '4': '#922b21', '5': '#1a5276', '6': '#7d3c98'}

# =====================================================================
# 绘图
# =====================================================================
fig, ax = plt.subplots(figsize=(18, 8))

ax.plot(x, df3['comm_pct'], marker='v', markersize=7, linewidth=2,
        color=color_fill['3'], label='3 MCUs')
ax.plot(x, df4['comm_pct'], marker='o', markersize=7, linewidth=2,
        color=color_fill['4'], label='4 MCUs')
ax.plot(x, df5['comm_pct'], marker='s', markersize=7, linewidth=2,
        color=color_fill['5'], label='5 MCUs')
ax.plot(x, df6['comm_pct'], marker='^', markersize=7, linewidth=2,
        color=color_fill['6'], label='6 MCUs')

# 50% 参考线
ax.axhline(y=50, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
ax.text(len(df4) - 0.5, 51, '50%', fontsize=9, color='gray', ha='right')

# 汇总统计框
avg3 = df3['comm_pct'].mean()
avg4 = df4['comm_pct'].mean()
avg5 = df5['comm_pct'].mean()
avg6 = df6['comm_pct'].mean()
total_comm3 = df3['comm'].sum()
total_comm4 = df4['comm'].sum()
total_comm5 = df5['comm'].sum()
total_comm6 = df6['comm'].sum()
summary = (
    f"Avg Comm. Ratio:\n"
    f"  3 MCU: {avg3:.1f}%  (total {total_comm3:.1f} ms)\n"
    f"  4 MCU: {avg4:.1f}%  (total {total_comm4:.1f} ms)\n"
    f"  5 MCU: {avg5:.1f}%  (total {total_comm5:.1f} ms)\n"
    f"  6 MCU: {avg6:.1f}%  (total {total_comm6:.1f} ms)"
)
ax.text(0.98, 0.95, summary, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fdfefe', alpha=0.95, edgecolor='#bdc3c7'),
        family='monospace')

ax.set_xlabel('Block Partition (Layer Range)', fontsize=11)
ax.set_ylabel('Comm. / Total Latency (%)', fontsize=11)
ax.set_ylim(0, 100)
ax.set_title('DSCNN Block Mode: Communication Ratio',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('./test/dscnn/dscnn_block_ratio.png', dpi=300)
# print("Saved to ./test/mbv2_035_block_ratio.png")
