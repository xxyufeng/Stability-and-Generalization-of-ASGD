# Stability-and-Generalization-of-ASGD
This repository is the implementation of the section "Experimental Results" of paper "Sharper Generalization Guarantees for Asynchronous SGD: Beyond Lipschitzness, Smoothness and Data Homogeneity"

---

## 📂 File Descriptions

### 1. `ASGD_Stab_Gen.py` (Stability Analysis)
This script focuses on analyzing the **Algorithmic Stability** of ASGD protocols. 
* **Mechanism:** It trains multiple models concurrently ($N+1$ models by default) on neighboring datasets (datasets differing by exactly one data point). 
* **Data Assignment:** The "changed" samples are deliberately assigned to specific workers (e.g., the fastest worker) to track how single-sample changes propagate in the asynchronous environment.
* **Key Metric:** Generalization Stability (measured as the L2 distance between the parameters of the reference model and neighboring models).

### 2. `ASGD_Gen_Heter.py` (Non-IID Performance)
This script focuses on the standard distributed learning setup under highly skewed non-IID conditions.
* **Mechanism:** Simulates severe data heterogeneity and trains a single model asynchronously.
* **Data Assignment:** Distributes the dataset entirely by `Label`. Each worker receives data belonging to a specific class (or a subset of classes), causing high variance among worker gradients.
* **Key Metrics:** Training/Test Loss, Generalization Gap.

### 3. `dataset_and_model.py` (Data & Model Utilities)
This script serves as the core utility module for all data loading, preprocessing, and model architecture definitions used across the ASGD experiments.

*   **Datasets Supported:** 
    *   **Vision Datasets:** MNIST, CIFAR-10.
    *   **LibSVM Datasets (Sparse/Tabular):** RCV1, GISETTE, a1a, w1a, ijcnn.
*   **Models Supported:**
    *   **Linear/Convex Models:** Linear Classifiers for RCV1, GISETTE, MNIST, CIFAR-10, a1a, w1a, ijcnn (`Linear_RCV1`, `Linear_MNIST`, etc.).
    *   **Non-Convex Neural Networks:** Multilayer Perceptrons (`FCNET_MNIST`), and small Deep CNNs adapted for CIFAR-10 (`ResNet18_CIFAR10`, `MobileNetV1_CIFAR10`, `ShuffleNetV2_CIFAR10`, `ResNet20_CIFAR10`).
    *   **Loss Functions:** Wraps training targets automatically using Mean Squared Error (`mse`), Cross Entropy (`ce`), or parameterized Hinge Loss (`hingeloss`).

---

## ⚙️ Core Arguments

When executing the baseline scripts (`ASGD_Stab_Gen.py` or `ASGD_Gen_Heter.py`), the following arguments dictate the experiment's behavior:

### Data & Model Configuration
*   `--dataset`: Name of the dataset to run (e.g., `mnist`, `cifar10`, `rcv1`, `w1a`).
*   `--dataset-path`: Destination path for downloading and parsing datasets. Default is `./data`.
*   `--model`: Name of the model mapping to `dataset_and_model.py` string literals (e.g., `fcnet_mnist`, `resnet20_cifar10`, `linear_rcv1`).
*   `--loss`: Loss function choice. Options: `mse`, `ce`, `hingeloss`.
*   `--q`: Parameter for q-norm hinge loss ( (q-1,L)-Holder continous, $\forall q\in [1,2]$ ). if `hingeloss` is selected (e.g., `q=1.5` for parameterized hinge loss). 

### Training & Optimizer Setup
*   `--lr`: Base Learning rate for the SGD optimizer.
*   `--lr-type`: `constant` for fixed learning rate, or `adaptive` which scales the learning rate down linearly proportional to the calculated delay of the gradient.
*   `--ada-fac`: Decay factor coefficient used only when `--lr-type=adaptive` is active.
*   `--batch-size`: Mini-batch size processed by each worker locally (Default is often `1` to simulate pure stochastic algorithms).
*   `--iterations`: Total number of global parameter updates the server actor will perform before terminating.
*   `--repeats`: Number of times the entire experiment runs with different random seeds.

### Distributed System Configuration
*   `--ASGD-type`: Specifies the distributed scheduling protocol. 
    *   `pure`: Vanilla Asynchronous update (no synchronization barriers).
    *   `random`: Random worker pulls scheme.
    *   `shuffle`: Shuffle worker pulls scheme.
    *   `waiting` / `random_waiting` / `shuffle_waiting`: Parameter Server waits for a specific bounded count of gradients before doing an averaged update step.
*   `--num-workers-list`: Specify single or multiple parallel worker capacities to test (e.g., `"1,4,8,16"` tests the system independently over 1, 4, 8, and 16 worker setups).
*   `--wait-b`: Defines the batch buffer constraint for algorithms initialized with a `waiting` flag.
*   `--slow-delay` / `--slow-delay-list`: Simulates stragglers by sleeping for $X$ seconds forcefully in slower workers.
*   `--n-pairs`: Controls $N$, the number of neighboring substituted datasets analyzed simultaneously for theoretical algorithmic stability evaluations. 

---

## 🚀 How to Run

**1. Install Dependencies:**

!pip install torch ray numpy pandas matplotlib

**2. Running the Stability Experiment:**

!python ASGD_Stab_Gen.py --ASGD-type pure --wait-b 1 --dataset mnist --model fcnet_mnist --loss ce --lr 5e-4 --lr-type constant --iterations 20000 --eval-interval 2000 --batch-size 4 --n-pairs 5 --num-samples 200 --num-workers-list 1,2,4,8,16,32 --repeats 1 --seed-base 43 --device cuda

**3. Running the Non-IID Experiment:**

!python ASGD_Gen_Heter.py --ASGD-type waiting --wait-b 4 --dataset mnist --model linear_mnist --loss ce --lr 1e-3 --lr-type constant --iterations 10000 --eval-interval 2000 --batch-size 4 --num-samples 200 --num-workers-list 10 --slow-delay-list 0,0.05,0.15,0.25,0.35,0.45,0.55 --repeats 1--device cuda
