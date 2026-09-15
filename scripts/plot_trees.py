#!/usr/bin/env python3
"""Plot the separate tree exploration and confirmation from tracked summaries."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
fig,ax=plt.subplots(figsize=(7.4,4.2))
for offset,folder,label,marker in [
    (-.08,'trees','Exploratory study: 2,048 seeds/family','o'),
    (.08,'trees_confirmation','Independent confirmation: 4,096 seeds/family','s')]:
    rows=json.loads((ROOT/'results'/folder/'summary.json').read_text())['primary']
    means=np.array([r['mean'] for r in rows])*100
    lows=np.array([r['lower'] for r in rows])*100
    highs=np.array([r['upper'] for r in rows])*100
    ax.errorbar(np.arange(3)+offset,means,yerr=[means-lows,highs-means],
                fmt=marker,capsize=5,label=label)
ax.axhline(0,linewidth=1,linestyle='--')
ax.set_xticks(np.arange(3),['Mixed','XOR-heavy','Logic-heavy'])
ax.set_ylabel('Recursion dividend (percentage points)')
ax.set_title('Better improvers, but worse recursive inheritance')
ax.legend(loc='lower right',fontsize=9)
ax.text(.02,.98,'Simultaneous finite-sample 95% intervals within each study',
        transform=ax.transAxes,va='top',fontsize=9)
fig.tight_layout()
out=ROOT/'paper/figures';out.mkdir(exist_ok=True,parents=True)
fig.savefig(out/'tree_dividend.pdf',bbox_inches='tight')
fig.savefig(out/'tree_dividend.png',dpi=180,bbox_inches='tight')
plt.close(fig)
