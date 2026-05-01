import matplotlib.pyplot as plt

clusters = {
    "Cluster 4": ["Ectopic preg.", "Pulmonary insuff.", "Resp. disease"],
    "Cluster 6": ["Oral inflam.", "GI inflammation"],
    "Cluster 8": ["Genital prolapse", "Pelvic inflam.", "PID severe"],
    "Cluster 26": ["Valve disease", "Chronic lung", "Pulm. failure"],
    "Cluster 27": ["Neuropathy", "Postpartum", "Pelvic inflam."],
    "Cluster 34": ["Salivary", "Cranial nerve", "Parkinsonism", "Ischemic heart"],
    "Cluster 38": ["Psych", "Mood disorder", "Ectopic preg."],
    "Cluster 209": ["Stroke", "Resp. comp.", "Upper resp."],
    "Cluster 427": ["Prostate mild", "Progression", "Severe"],
    "Cluster 842": ["GI symptoms", "GI diagnosis"],
    "Cluster 1698": ["Constipation", "Alt. bowel", "Anal disorder"],
    "Cluster 1702": ["Breast benign", "Preg. loss", "Complication"]
}

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(14, 9))

x_spacing = 2.0
box_pad = 0.35

for i, (cluster_name, diseases) in enumerate(clusters.items()):
    y = len(clusters) - i

    # cluster label (left aligned, larger, better spacing)
    ax.text(
        -1.2, y,
        cluster_name,
        ha='right', va='center',
        fontsize=13,
        fontweight='bold'
    )

    prev_x = None

    for j, disease in enumerate(diseases):
        x = j * x_spacing

        # disease node
        ax.text(
            x, y, disease,
            ha='center', va='center',
            fontsize=12,
            bbox=dict(
                boxstyle="round,pad=0.4",
                fc="white",
                ec="black",
                lw=1.2
            ),
            zorder=3   # ensure boxes are on top
        )

        # arrows BETWEEN boxes (drawn underneath text)
        if prev_x is not None:
            ax.annotate(
                "",
                xy=(x - box_pad, y),
                xytext=(prev_x + box_pad, y),
                arrowprops=dict(
                    arrowstyle="-|>",
                    lw=2.2,
                    color="black",
                    mutation_scale=18
                ),
                zorder=1   # push arrows behind boxes
            )

        prev_x = x

# dynamic limits (important for varying cluster sizes)
max_len = max(len(v) for v in clusters.values())

ax.set_xlim(-2, (max_len - 1) * x_spacing + 1)
ax.set_ylim(0, len(clusters) + 1)

ax.axis("off")

plt.title("Temporal Disease Trajectories Across Clusters", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('Temporal.png', dpi=150)
plt.show()