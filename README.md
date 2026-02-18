# Learning Ordinal–Hierarchical Constraints for Deep Learning Classifiers

This repository contains the official implementation of the methodologies described in the paper:
"Learning Ordinal–Hierarchical Constraints for Deep Learning Classifiers" Published in IEEE Transactions on Neural Networks and Learning Systems (2024).

---

Standard Deep Learning (DL) classifiers often struggle with datasets that exhibit both hierarchical and ordinal relationships. This research introduces a framework to leverage these constraints simultaneously, significantly improving generalization and prediction consistency.

We propose two novel methodologies:

- Hierarchical Cumulative Link Model (HCLM): Integrates the ordinal cumulative link model within a hierarchical structure.

- Hierarchical–Ordinal Binary Decomposition (HOBD): Decomposes the problem into local and global graph paths, minimizing joint losses.

The effectiveness of these models is demonstrated across various domains, including industrial quality control, biomedical analysis, financial forecasting, and computer vision.

Key Features:

* Dual Constraint Integration: Seamlessly combines ordinal ranking with hierarchical classification.

* Flexible Loss Functions: Implementations of HCLM and HOBD that can be plugged into most standard DL architectures.

* Cross-Domain Performance: Validated on diverse real-world datasets.

* Consistency-Driven: Ensures that predictions at the parent level are logically consistent with child-level ordinal rankings.
