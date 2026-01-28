# MorphSNN: Adaptive Graph Diffusion and Structural Plasticity for Spiking Neural Networks

---

**Table of contents:**

- [Abstract](#abstract)
- [Dependency](#dependency)
- [Directory Tree](#directory)
- [Usage](#usage)

## Abstract

 Spiking Neural Networks (SNNs) currently face a critical bottleneck: while individual neurons exhibit dynamic biological properties, their macroscopic architectures remain confined within rigid, static connectivity patterns. This mismatch between neuron-level dynamics and network-level fixed connectivity deprives networks of critical lateral interaction mechanisms inherent in the brain, hindering adaptive capacity in dynamic environments.  To address this, we propose MorphSNN, a backbone framework inspired by biological non-synaptic diffusion and structural plasticity. Specifically, we introduce a \textbf{Graph Diffusion (GD)} mechanism to facilitate efficient undirected signal propagation, complementing the feedforward hierarchy. Furthermore, it incorporates a \textbf{Spatio-Temporal Structural Plasticity (STSP)} mechanism, endowing the network with the capability for instance-specific, dynamic topological reorganization, thereby overcoming the limitations of fixed topologies. Experiments demonstrate that MorphSNN achieves State-of-the-Art accuracy on static and neuromorphic datasets; for instance, it reaches \textbf{83.35\%} accuracy on N-Caltech101 with only 5 timesteps. More importantly, its self-evolving topology functions as an intrinsic distribution fingerprint, enabling superior Out-of-Distribution (OOD) detection without auxiliary training.  
## Dependency

The major dependencies of this repo are listed as below.

```
# Name                 Version
python                  3.9.19
torch                   2.0.0
torchvision             0.15.0
spikingjelly            0.0.0.0.14
torchinfo				1.8.0
tqdm					4.66.4
matplotlib				3.10.8
networkx				3.2.1
numpy					2.4.1
scikit_learn			1.8.0
seaborn					0.13.2
thop					0.1.1.post2209072238
timm					0.6.12
```

## Directory Tree

```
|-- Classification_neuromorphic_data/ 
	|--CIFAR10-DVS  
		|--Full		# Full model
		|--Static	# Static variant
	|--DVS-Gesture  
		|--Full
		|--Static
	|--NCaltech101
		|--Full
		|--Static
	|--UCF101-DVS
|-- Classification_static_data
	|--CIFAR10
	|--CIFAR100
	|--ImageNet
|-- MorphANN
	|--CIFAR10
	|--CIFAR100
|-- OoD 
|-- Robustness

```

## Usage

1. **DVS-Gesture Training**

    In DVS Gesture, we provide two variants, Full and Static, with the optimal performance achieved through train.py in Full.

    ```bash
    cd Classification_neuromorphic_data/DVS-Gesture/Full
    python train.py
    ```

2. **CIFAR10-DVS Training**

    The optimal performance of CIFAR10-DVS is achieved by the static variant, so

    ```bash
    cd Classification_neuromorphic_data/CIFAR10-DVS/Static
    python train.py
    ```

3. **NCaltech101 Training**

    The optimal performance of NCaltech101 is achieved by the static variant, so

    ```bash
    cd Classification_neuromorphic_data/NCaltech101/Static
    python train.py
    ```

4. **UCF101-DVS  Training**

    You need to download the UCF101-DVS dataset by yourself, which is 28.4GB in size. Due to time constraints, I cannot find the download link at the moment. I will update the download link promptly after the review is completed.

    The dataset needs preprocessing, so the command flow is

    ```bash
    cd Classification_neuromorphic_data/UCF101DVS
    python preprocess.py
    python train.py
    ```

5. **CIFAR10 Training**

    The optimal performance of CIFAR10 is achieved by the static variant, so

    ```bash
    cd Classification_static_data/CIFAR10/Static
    python train.py
    ```

6. **CIFAR100 Training**

    The optimal performance of CIFAR100 is achieved by the static variant, so

    ```bash
    cd Classification_static_data/CIFAR100/Static
    python train.py
    ```

7. **ImageNet Training**

   The training script for ImageNet can refer to [QKFormer](https://github.com/zhouchenlin2096/QKFormer), just replace our model with it.

   ```bash
   cd Classification_static_data/CIFAR100/Static
   python smodel.py
   ```

8. **CIFAR10 Training by MorphANN**

   This is an ANN variant that only integrates the GD mechanism because there is no timestep

   ```bash
   cd MorphANN/CIFAR10
   python train.py
   ```

9. **CIFAR100 Training by MorphANN**

   ```bash
   cd MorphANN/CIFAR100
   python train.py
   ```

10. **OoD Detection**

    In OoD/IN DVSGesture, we provide the optimal model weights for training MorphSNN on the DVSGesture dataset - best_madel.ablation-psh.
    In addition, DGP.py, kNN.py, MSP.py, SCP.py, Energy.py are provided. Simply change OoD_DATASET="CIFAR10-DVS" and the corresponding directory. The acquisition of the DVS Lip dataset can refer to the [link](https://drive.google.com/file/d/1dBEgtmctTTWJlWnuWxFtk8gfOdVVpkQ0/view

    ```bash
    cd OoD/ID-DVSGesture
    python DGP
    ```

11.  **Robustness**

    The robustness of the proposed STSP mechanism is evaluated by injecting different types of perturbations during testing.
    The implementation is provided in `test.py` under `Robustness/Full` and `Robustness/Static`.
    Different noise models can be enabled by simply uncommenting the corresponding line:

    - Salt-and-pepper noise in line 123: `add_sp`
    - Poisson noise in line 125: `add_pn`
    - Row-drop noise in line 128: `row_drop_ratio`



The training scripts for CIFAR10, CIFAR100, and ImageNet are from [QKFormer](https://github.com/zhouchenlin2096/QKFormer) .

Thank them for providing a unified script for SNN on static datasets

