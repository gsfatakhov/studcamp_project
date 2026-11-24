# GNN for fault diagnosis on the Tennessee Eastman Process

Project by AIRI at the Math in AI summer camp (Yandex & Innopolis University, July 2024).

Team: Georgii Fatakhov, Alexey Yanin, German Kuznetsov.

The model is based on the approach from Kovalenko, Pozdnyakov, Makarov, [Graph Neural Networks with Trainable Adjacency Matrices for Fault Diagnosis on Multivariate Sensor Data](https://arxiv.org/abs/2210.11164) (AIRI).

The task is fault detection and diagnosis on multivariate sensor data from the Tennessee Eastman Process (TEP), a standard benchmark of a simulated chemical plant. Each sample is a window of readings from 52 sensors, and the model classifies the state of the process: normal operation or one of the faults.

## Approach

Each sensor is a node of a graph. The model is a graph neural network with a graph structure learning layer, which builds the adjacency matrix used by the graph convolutions. Several variants of this layer are implemented in `layers.py`:

- `relu`, `directed`, `unidirected`, `undirected`: the adjacency matrix is learned during training;
- `constant`: a fixed adjacency matrix built from expert knowledge of the process (which sensors and manipulated variables are physically connected), see `get_expert_matrix()` in the notebook.

## Files

- `layers.py`: graph convolution layer and graph structure learning layers
- `models.py`: the `GNNTEP` model
- `gnn_tep.ipynb`: data loading, training and evaluation

## Data and evaluation

Datasets and metrics come from [fddbenchmark](https://github.com/airi-industrial-ai/fddbenchmark) (`small_tep`, `rieth_tep`, `reinartz_tep`):

```
pip install git+https://github.com/airi-industrial-ai/fddbenchmark
```

The notebook reports accuracy and detection metrics (TPR, FPR, average detection delay).
