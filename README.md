# COMS 6998 Practical Deep Learning Systems Performance Final Project -- Federated Learning
Sankalp Apharande (spa2138)

Katie Jooyoung Kim (jk4534)

### Project Description
In our final project for the course Practical Deep Learning Systems Performance, we build upon our mid-term seminar presentation on the topic of federated learning. To this end, we delve further into the following topics. 

1. Federated Learning with Only Positive Labels [Paper Here](https://arxiv.org/abs/2004.10342)

Federated learning with only positive labels refers to a specific yet realistic setting in which each client participating in the federated training process only has access to local data from a specific class. This is an issue because the overall problem is still that of a multi-class classification scheme. 

### Repository Description
- generate_most_frequent_labels.ipynb: 
- most_frequent_labels.json: results obtained from generate_most_frequent_labels.ipynb
- FederatedAveraging.ipynb: MNIST classification task using the FederatedAveraging algorithm proposed by the [original paper that introduced federated learning](https://arxiv.org/abs/1602.05629)
- FederatedAveragingwithSpreadout.ipynb: MNIST classification task using the FederatedAveragingwithSpreadout algorithm designed to address the issues inherent to classification in an "only positive labels" setting

### Example Commands
N/A (straightforward for Jupyter Notebooks)

### Results and Observations

In both of the experiments, we observe that federated learning is very slow in speed and in convergence compared to similar tasks that we have encountered in the course. This is true for both the frameworks that we have used (PySyft and PyTorch / TensorFlow Federated and TensorFlow). In the MNIST image classification task, 

