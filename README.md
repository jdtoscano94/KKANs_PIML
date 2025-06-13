# Kurkova-Kolmogorov-Arnold-Newtorks-KKANs

Inspired by the Kolmogorov-Arnold representation theorem and Kurkova's principle of using approximate representations, we propose the Kurkova-Kolmogorov-Arnold Network (KKAN), a new two-block architecture that combines robust multi-layer perceptron (MLP) based inner functions with flexible linear combinations of basis functions as outer functions. We first prove that  KKAN is a universal approximator, and then we demonstrate its versatility across scientific machine-learning applications, including function regression, physics-informed machine learning (PIML), and operator-learning frameworks. The benchmark results show that KKANs outperform MLPs and the original Kolmogorov-Arnold Networks (KANs) in function approximation and operator learning tasks and achieve performance comparable to fully optimized MLPs for PIML. To better understand the behavior of the new representation models, we analyze their geometric complexity and learning dynamics using information bottleneck theory, identifying three universal learning stages, fitting, transition, and diffusion, across all types of architectures. We find a strong correlation between geometric complexity and signal-to-noise ratio (SNR), with optimal generalization achieved during the diffusion stage. Additionally, we propose self-scaled residual-based attention weights to maintain high SNR dynamically, ensuring uniform convergence and prolonged learning. 


## Papers

Currently, this repository contains the code for the following paper:

1. **Toscano, J. D., Wang, L. L., & Karniadakis, G. E.** (2024). KKANs: Kurkova-Kolmogorov-Arnold Networks and Their Learning Dynamics. arXiv preprint arXiv:2412.16738.

If you find this content useful please consider citing our work as follows:

```bibtex
@article{toscano2024kkans,
  title={KKANs: Kurkova-Kolmogorov-Arnold Networks and Their Learning Dynamics},
  author={Toscano, Juan Diego and Wang, Li-Lian and Karniadakis, George Em},
  journal={arXiv preprint arXiv:2412.16738},
  year={2024}
}
```

## Prerequisites

* **Python**: Recommended version is ≥ 3.9.
* **NVIDIA GPU**: Required to run the models, along with a compatible CUDA toolkit.
* **Conda**: Recommended for managing Python environments and dependencies.
* **Important**: The code is compatible with any version of JAX ≥ 0.3.23 and any version of NumPy. 

## Critical Step


JAX is a critical component for this project. Correct installation, which depends on your system's NVIDIA driver and CUDA toolkit version, is essential.

**Crucial First Step:** Always consult the **[official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html#nvidia-gpu)**. The commands provided there are the most current and specific to different CUDA versions.

* **Standard Pip Command (Recommended starting point - Example for CUDA 12):**
    The official JAX guide will provide a command similar to the following (ensure you use the exact command appropriate for your CUDA version):
    ```bash
    pip install -U "jax[cuda12_pip]"
    ```
    *(Note: The `cuda12_pip` part is specific to CUDA 12. If you use a different CUDA version, like CUDA 11, you'll need to use the corresponding identifier, e.g., `cuda11_pip`. This command installs JAX, JAXlib, and the necessary NVIDIA libraries for JAX, all managed via pip.)*


## Instructions

1. **Clone the repository**:
   ```sh
   git clone https://github.com/jdtoscano94/Kurkova-Kolmogorov-Arnold-Newtorks-KKANs.git
   cd YOUR_REPO_FOLDER
   ```

2. **Download the dataset**:  
   The dataset for the Operator learnign tasks is available [here](https://drive.google.com/drive/folders/1zLwn4IqnmV0d4ELTEIal4cjssI3FuXv-?usp=drive_link).

3. **Move the data to the correct directory**:  
   ```sh
   mv path_to_downloaded_data ../Data/
   ```

4. **Run our models using the provided Jupyter notebooks and python files**.  
   The repository includes results for function approximation, physics-informed machine learnign and Operator learning using:
   - **MLPs** 
   - **cKANs** 
   - **KKANs** 

   Each file contains all the necessary code to **replicate the results** presented in the paper. The results were generated using **JAX**.

**Note:** To run these files, you need the source files located in `../Crunch`. These files should be automatically downloaded when the project is cloned. Ensure that all dependencies are installed before executing the notebooks.

## References

If you use this repository in your research, please cite our related work:

```bibtex
@article{anagnostopoulos2024residual,
  title={Residual-based attention in physics-informed neural networks},
  author={Anagnostopoulos, Sokratis J and Toscano, Juan Diego and Stergiopulos, Nikolaos and Karniadakis, George Em},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={421},
  pages={116805},
  year={2024},
  publisher={Elsevier}
}

@article{anagnostopoulos2024learning,
  title={Learning in PINNs: Phase transition, total diffusion, and generalization},
  author={Anagnostopoulos, Sokratis J and Toscano, Juan Diego and Stergiopulos, Nikolaos and Karniadakis, George Em},
  journal={arXiv preprint arXiv:2403.18494},
  year={2024}
}

@article{shukla2024comprehensive,
  title={A comprehensive and fair comparison between mlp and kan representations for differential equations and operator networks},
  author={Shukla, Khemraj and Toscano, Juan Diego and Wang, Zhicheng and Zou, Zongren and Karniadakis, George Em},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={431},
  pages={117290},
  year={2024},
  publisher={Elsevier}
}

@article{toscano2024inferring,
  title={Inferring in vivo murine cerebrospinal fluid flow using artificial intelligence velocimetry with moving boundaries and uncertainty quantification},
  author={Toscano, Juan Diego and Wu, Chenxi and Ladr{\'o}n-de-Guevara, Antonio and Du, Ting and Nedergaard, Maiken and Kelley, Douglas H and Karniadakis, George Em and Boster, Kimberly AS},
  journal={Interface Focus},
  volume={14},
  number={6},
  pages={20240030},
  year={2024},
  publisher={The Royal Society}
}

@article{toscano2025pinns,
  title={From pinns to pikans: Recent advances in physics-informed machine learning},
  author={Toscano, Juan Diego and Oommen, Vivek and Varghese, Alan John and Zou, Zongren and Ahmadi Daryakenari, Nazanin and Wu, Chenxi and Karniadakis, George Em},
  journal={Machine Learning for Computational Science and Engineering},
  volume={1},
  number={1},
  pages={1--43},
  year={2025},
  publisher={Springer}
}


@article{toscano2025aivt,
  title={AIVT: Inference of turbulent thermal convection from measured 3D velocity data by physics-informed Kolmogorov-Arnold networks},
  author={Toscano, Juan Diego and K{\"a}ufer, Theo and Wang, Zhibo and Maxey, Martin and Cierpka, Christian and Karniadakis, George Em},
  journal={Science Advances},
  volume={11},
  number={19},
  year={2025},
  month={May},
  day={7},
  doi={10.1126/sciadv.ads5236},
  URL={https://www.science.org/doi/10.1126/sciadv.ads5236}
}

@article{wu2025fmenets,
  title={FMEnets: Flow, Material, and Energy networks for non-ideal plug flow reactor design},
  author={Wu, Chenxi and Toscano, Juan Diego and Shukla, Khemraj and Chen, Yingjie and Shahmohammadi, Ali and Raymond, Edward and Toupy, Thomas and Nazemifard, Neda and Papageorgiou, Charles and Karniadakis, George Em},
  journal={arXiv preprint arXiv:2505.20300},
  year={2025}
}

```
