## Evolving Disease Progression Networks through Incremental Refinement

### Genie.py.
<p align="justify">
GENIE3 is a tree-based ensemble method that infers a gene regulatory network by treating each gene as a target and learning how well all other genes (potential regulators) can predict its expression using Random Forest or Extra-Trees models. The resulting feature importances from all target-specific models are aggregated into a weighted adjacency matrix, where each entry represents a directed regulatory influence score between gene pairs.
</p>

### Read_Diagnosis.py.
<p align="justify">
This code preprocesses the MIMIC-3 diagnosis dataset into a patient–disease binary incidence matrix, where rows represent patients and columns represent ICD9-coded diseases. It filters out inactive or uninformative disease columns and constructs a clean input representation suitable for downstream modeling. The processed matrix is then fed into GENIE3 to infer a weighted disease–disease interaction network based on co-occurrence structure across patients. The resulting network captures latent dependencies among diseases and is saved for subsequent trajectory modeling and clustering analyses.
</p>

### Cond.py.
<p align="justify">
This code analyzes disease–disease relationships by combining clinical co-occurrence structure with a learned GENIE network. It clusters diseases using hierarchical clustering on a symmetrized gene–gene interaction matrix (GENIE weights), and visualizes the resulting hierarchy as a dendrogram. In parallel, it constructs empirical disease co-occurrence probabilities from diagnosis data and maps ICD9 codes into coarse categories for higher-level clinical grouping. Finally, it evaluates how well the learned GENIE edge weights align with real-world conditional disease associations by binning edges by weight and computing the average conditional probability within each bin. The resulting plot shows whether stronger inferred GENIE connections correspond to higher observed clinical co-occurrence, providing a validation of the network’s biological and clinical interpretability.
</p>

### Heatmap.py.
<p align="justify">
This pipeline performs hierarchical clustering on a GENIE3-derived gene–disease association matrix after symmetrizing and converting it into a distance space via an exponential transform. It then evaluates cluster quality by mapping ICD9-based disease groupings onto clusters, computing concordance between biological taxonomy and learned structure, and visualizing cluster-wise ICD9 enrichment via heatmaps.
</p>

### gmlGraph_kneePlot.py; kneePoint.py.
<p align="justify">
The Kneedle algorithm is used to automatically detect the “elbow point” in the sorted distribution of GENIE edge weights, where the curve transitions from steep (informative, high-confidence edges) to flat (noisy or low-signal edges). This point is identified by measuring the maximum deviation between the normalized weight curve and a reference diagonal, effectively capturing where marginal gains in retained structure begin to diminish. By thresholding at this elbow, the GENIE network can be sparsified in a data-driven way that preserves structurally meaningful edges while removing weak, less informative connections.
</p>

### Adjust_Main.py.
<p align="justify">
This code simulates an iterative learning framework for a disease progression network (GENIE-style model) using synthetic patient trajectories generated as random walks on a weighted directed graph. At each iteration, it constructs patient-level transition frequencies from simulated disease sequences, aggregates them into a temporary observation network, and updates the underlying disease graph by reinforcing frequently observed transitions while conserving total edge weight through compensatory down-weighting of unobserved edges. The model thereby performs a dynamic, data-driven refinement of disease–disease relationships based on simulated longitudinal patient behavior. Over repeated updates, the algorithm tracks convergence between the evolving learned network and the original ground-truth structure using Kendall’s Tau and Pearson correlation, providing quantitative measures of structural recovery. This allows evaluation of how well the iterative update mechanism reconstructs the underlying disease transition dynamics from trajectory data, and how parameters such as update rate (delta) and decay influence stability and convergence behavior.
</p>

### Viz.py; Temporal_plot.py.
<p align="justify">
The visualization displays disease trajectories organized into clusters, where each row represents a clinically meaningful disease subtype inferred from GENIE-based network analysis. Within each cluster, diseases are arranged sequentially and connected with directed arrows to illustrate typical temporal progression patterns observed across pseudo-patient trajectories. Node boxes are rendered with clear spacing, high-contrast styling, and controlled layout to ensure interpretability even in dense clusters. Arrows are intentionally drawn beneath the boxes to preserve readability while still emphasizing directional disease evolution within each subtype.
</p>

### Parse.py; Naming.py.
<p align="justify">
This code constructs an ICD-9 category mapping by parsing a tab-separated range-to-label file and serializing it for later use in disease ontology alignment. It then loads GENIE-derived gene/disease graphs, compares the original and trimmed networks using node-label mappings, and identifies edges that are present in the trimmed graph but absent in the original GENIE network.
</p>
