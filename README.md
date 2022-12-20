# COMS 6998 Practical Deep Learning Systems Performance Final Project
## A Comparative Study of Federated Learning


Repository for Practical Deep Learning System Performance Fall 2022 Project: A Comparative Study of Federated Learning Federated Learning


**Contributors:**
1.  Sankalp Apharande, Columbia University, New York, NY 10027 (Email: spa2138@columbia.edu)

2.  Katie Jooyoung Kim, Columbia University, New York, NY 10027 (Email: jk4534@columbia.edu)

### Project Description
In our final project for the course Practical Deep Learning Systems Performance, we build upon our mid-term seminar presentation on the topic of federated learning. To this end, we delve further into the following topics. 
We conduct a comparative study of the performance of federated learning vs. traditional deep learning.
Federated learning is a technique that allows multiple participating agents (“clients”) to learn an aggregate model that is coordinated by a central server without sending their sensitive data to the latter. 
Federated optimization differs from distributed optimization in that it takes the distributed nature of the operations one step further by keeping the data itself distributed as well.
Our goal is to demonstrate how federated learning is a powerful way to realize decentralized training tasks through using it in a practical setting and conducting a comparative study. 

**Related Topics**
1. Federated Learning with Only Positive Labels [Paper Here](https://arxiv.org/abs/2004.10342)
Federated learning with only positive labels refers to a specific yet realistic setting in which each client participating in the federated training process only has access to local data from a specific class. This is an issue because the overall problem is still that of a multi-class classification scheme. 
2. Federated Learning for Mobile Keyboard Prediction [Paper Here](https://arxiv.org/abs/1811.03604) 
Federated learning for next word prediction

### Results and Observations
In both of the experiments, we observe that federated learning is very slow in speed and in convergence compared to similar tasks that we have encountered in the course. This is true for both the frameworks that we have used (PySyft and PyTorch / TensorFlow Federated and TensorFlow). In the MNIST image classification task, 


**Approach:**
We conduct 2 experiments:
1. Multi-Class Classification with Only Positive Labels
2. Federated Learning for Next word Prediction

***Multi-Class Classification with Only Positive Labels:***
1. Server-based training will be conducted via the Google Cloud Platform Compute Engine Service using a NVIDIA Tesla P4 GPU
2. To conduct federated learning with ease, we use the TensorFlow Federated library (0.38.0) from Google. 
3. We use the MNIST handwritten digits dataset to simulate the actual constraints of a “multi-class classification with only positive labels” scheme. 

***Federated Learning for Next word Prediction:***
1. First we trained BiLSTM Model for next word prediction on sentiment140 dataset containing 1.6M tweets
2. Then we used the pretrained model to start federated training on remote and local datasets of each user

We also deployed server trained model to custom vertex AI endpoint for Latency Load and throughput analysis

**A description of the repository:**
1. `next_word_prediction/server_based_BiLSTM.ipynb` : Training and evaluation of serverbased BiLSTM Model
2. `next_word_prediction/federated`: Federated Training
   1. `next_word_prediction/federated/data`: processed data, generated word to index mapping and trained lstm model from above approach
   2. `next_word_prediction/federated/utils`: utils files for constants and model architecture
   3. `next_word_prediction/federated/main.py`: main file to trigger federated training
3. `practical-dl-prediction-endpoint`: Dockerfile and required code to preprocess incoming inference request and generate inference results
4. `generate_most_frequent_labels.ipynb` and `most_frequent_labels.json`: results obtained from generate_most_frequent_labels.ipynb 
5. `FederatedAveraging.ipynb`: MNIST classification task using the FederatedAveraging algorithm proposed by the [original paper that introduced federated learning](https://arxiv.org/abs/1602.05629)
6. `FederatedAveragingwithSpreadout.ipynb`: MNIST classification task using the FederatedAveragingwithSpreadout algorithm designed to address the issues inherent to classification in an "only positive labels" setting

**Steps to trigger server-based BiLSTM training:** 
1. Setup GCP VM with V100 GPU and PyTorch 1.4.0 with Jupyter Notebook access
2. Run `next_word_prediction/server_based_BiLSTM.ipynb` notebook step by step

**Steps to trigger Federated BiLSTM training:**
1. Launch High Memory VM on GCP
2. Download Trained LSTM model from this GDrive Link: `https://drive.google.com/file/d/14_I8zav51oatXySC0QNT8MIEV73dFBsV/`
3. SSH into that VM instance and create virtual environment with `next_word_prediction/federated/requirements.txt`: 
   1. `git clone <repo>`
   2. `cd <repo>/next_word_prediction/federated`
   3. `mv ~/LSTM_model_top3.pth ~/6998/next_word_prediction/federated/data/` 
   3. `pip install virtualenv`
   4. `virtualenv venv`
   5. `source venv/bin/activate`
   6. `pip install -r requirements.txt`
   7. Run `python3 main.py`

**Steps to Launch Vertex AI custom endpoint:**
1. Set up the following variables: <br>
— `PROJECT_ID=practical-dl` <br>
— `REGION=us-west1` <br>
— `REPO_NAME=6998` <br>
— `IMAGE_NAME=practical-dl-project` <br>
— `IMAGE_TAG=latest` <br>
2. `IMAGE_URI=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}`
3. `docker build -f Dockerfile -t ${IMAGE_URI} ./`
4. `gcloud beta artifacts repositories create $REPO_NAME \
 — repository-format=docker \
 — location=$REGION
docker push ${IMAGE_URI}`
5. Importing the model to VertexAI model registry and deploying that model to vertex ai endpoint were done from GCP console. Tutorial `https://cloud.google.com/vertex-ai/docs/predictions/get-predictions#google-cloud-console`



## Results: 
1. Only Positive Class: Results are shown in `FederatedAveraging.ipynb` and `FederatedAveragingwithSpreadout.ipynb` notebooks.

2. Next Word Prediction
   1. Server Based Training
![Alt text](/6998/next_word_prediction/top1_server.png?raw=true "Title")
   2. Federated Training:









