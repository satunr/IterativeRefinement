## Evolving Disease Progression Networks through Incremental Refinement

### Cond.py.
<p align="justify">
This code analyzes disease–disease relationships by combining clinical co-occurrence structure with a learned GENIE network. It clusters diseases using hierarchical clustering on a symmetrized gene–gene interaction matrix (GENIE weights), and visualizes the resulting hierarchy as a dendrogram. In parallel, it constructs empirical disease co-occurrence probabilities from diagnosis data and maps ICD9 codes into coarse categories for higher-level clinical grouping. Finally, it evaluates how well the learned GENIE edge weights align with real-world conditional disease associations by binning edges by weight and computing the average conditional probability within each bin. The resulting plot shows whether stronger inferred GENIE connections correspond to higher observed clinical co-occurrence, providing a validation of the network’s biological and clinical interpretability.
</p>

### gmlGraph_kneePlot.py; kneePoint.py.
<p align="justify">
The Kneedle algorithm is used to automatically detect the “elbow point” in the sorted distribution of GENIE edge weights, where the curve transitions from steep (informative, high-confidence edges) to flat (noisy or low-signal edges). This point is identified by measuring the maximum deviation between the normalized weight curve and a reference diagonal, effectively capturing where marginal gains in retained structure begin to diminish. By thresholding at this elbow, the GENIE network can be sparsified in a data-driven way that preserves structurally meaningful edges while removing weak, less informative connections.
</p>

### Adjust_Main.py.
<p align="justify">
This code simulates an iterative learning framework for a disease progression network (GENIE-style model) using synthetic patient trajectories generated as random walks on a weighted directed graph. At each iteration, it constructs patient-level transition frequencies from simulated disease sequences, aggregates them into a temporary observation network, and updates the underlying disease graph by reinforcing frequently observed transitions while conserving total edge weight through compensatory down-weighting of unobserved edges. The model thereby performs a dynamic, data-driven refinement of disease–disease relationships based on simulated longitudinal patient behavior. Over repeated updates, the algorithm tracks convergence between the evolving learned network and the original ground-truth structure using Kendall’s Tau and Pearson correlation, providing quantitative measures of structural recovery. This allows evaluation of how well the iterative update mechanism reconstructs the underlying disease transition dynamics from trajectory data, and how parameters such as update rate (delta) and decay influence stability and convergence behavior.
</p>
