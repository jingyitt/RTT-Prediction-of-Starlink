# RTT-Prediction-of-Starling

The model combines dynamic graph construction with temporal transformers to capture both spatial and temporal dependencies.

### 1. `build_graphs_from_tle.py`
**Graph Construction Module**
* **Function:** Processes Two-Line Element (TLE) data to construct dynamic graph structures.
* **Logic:** It establishes the topology based on the visibility and communication links between service satellites and target nodes.

### 2. `dgcn.py`
**Graph Feature Extraction**
* **Function:** Implements the Dynamic Graph Convolutional Network (DGCN) layer.
* **Logic:** Extracts high-level spatial and structural features from the graphs generated in the previous step, handling the changing topology of the satellite constellation.

### 3. `doubleLayer.py`
**Temporal Prediction Model**
* **Function:** The core prediction module using a Dual-Layer Temporal Transformer.
* **Logic:** Takes the sequence of graph features (output from DGCN) and applies attention mechanisms to capture long-term temporal dependencies for the final prediction task.

## Paper Reference
Original paper:
> Tian, Jingyi, Wenjun Yang, and Lin Cai. "Attention-Based Spatiotemporal Model for RTT Prediction in LEO Satellite Networks." ICC 2025 - IEEE International Conference on Communications.
