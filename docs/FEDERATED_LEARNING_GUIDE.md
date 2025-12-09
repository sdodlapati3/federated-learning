# Federated Learning: A Comprehensive Learning Guide

> **Purpose**: This document serves as a single entry point for understanding Federated Learning (FL) and related concepts including Differential Privacy (DP), distributed optimization, and privacy-preserving machine learning. Each section includes in-text citations and an extensive reading list.

---

## Table of Contents

1. [What is Federated Learning?](#1-what-is-federated-learning)
2. [Core Challenges in Federated Learning](#2-core-challenges-in-federated-learning)
3. [Federated Learning Algorithms](#3-federated-learning-algorithms)
4. [Differential Privacy Fundamentals](#4-differential-privacy-fundamentals)
5. [Differential Privacy in Federated Learning](#5-differential-privacy-in-federated-learning)
6. [DP Variants: Approximate, Rényi, Local, and Bayesian](#6-dp-variants-approximate-rényi-local-and-bayesian)
7. [Distributed Optimization Foundations](#7-distributed-optimization-foundations)
8. [Systems and Infrastructure](#8-systems-and-infrastructure)
9. [Advanced Topics](#9-advanced-topics)
10. [Practical Implementation Guide](#10-practical-implementation-guide)
11. [Master Reading List](#11-master-reading-list)

---

## 1. What is Federated Learning?

### 1.1 Definition

**Federated Learning (FL)** is a machine learning paradigm where multiple clients (devices, organizations, or data silos) collaboratively train a shared model while keeping their data decentralized (McMahan et al., 2017). Unlike traditional centralized learning, raw data never leaves the client's device—only model updates (gradients or weights) are shared with a central server.

The canonical formulation of FL solves the following optimization problem:

$$\min_{w} F(w) = \sum_{k=1}^{K} \frac{n_k}{n} F_k(w)$$

where:
- $K$ is the number of clients
- $n_k$ is the number of samples on client $k$
- $n = \sum_k n_k$ is the total number of samples
- $F_k(w)$ is the local empirical risk on client $k$

### 1.2 Taxonomy of Federated Learning

FL can be categorized along several dimensions (Yang et al., 2019; Kairouz et al., 2021):

| Dimension | Categories | Description |
|-----------|------------|-------------|
| **Data Partitioning** | Horizontal (sample-based) | Same features, different samples |
| | Vertical (feature-based) | Same samples, different features |
| | Federated Transfer | Different feature and sample spaces |
| **Scale** | Cross-device | Millions of mobile/IoT devices |
| | Cross-silo | Tens of organizations/data centers |
| **Aggregation** | Centralized | Single server coordinates |
| | Decentralized | Peer-to-peer communication |

### 1.3 Comparison with Related Paradigms

| Paradigm | Data Location | Model Training | Privacy |
|----------|---------------|----------------|---------|
| Centralized ML | Central server | Server | Low (data exposed) |
| Federated Learning | Distributed | Collaborative | Medium (updates exposed) |
| FL + DP | Distributed | Collaborative + noise | High |
| On-device Learning | Device only | Local only | Highest (no sharing) |

### 1.4 Real-World Applications

1. **Mobile Keyboard Prediction**: Google's Gboard uses FL to improve next-word prediction without uploading keystrokes (Hard et al., 2018)
2. **Healthcare**: Collaborative model training across hospitals while protecting patient data (Rieke et al., 2020)
3. **Financial Services**: Fraud detection across banks without sharing transaction data (Yang et al., 2019)
4. **Autonomous Vehicles**: Learning from distributed fleet data (Lim et al., 2020)

### References

- McMahan, B., Moore, E., Ramage, D., Hampson, S., & Arcas, B. A. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS. [[Paper]](https://arxiv.org/abs/1602.05629)
- Yang, Q., Liu, Y., Chen, T., & Tong, Y. (2019). *Federated Machine Learning: Concept and Applications*. ACM TIST. [[Paper]](https://arxiv.org/abs/1902.04885)
- Kairouz, P., et al. (2021). *Advances and Open Problems in Federated Learning*. Foundations and Trends in ML. [[Paper]](https://arxiv.org/abs/1912.04977)
- Hard, A., et al. (2018). *Federated Learning for Mobile Keyboard Prediction*. arXiv. [[Paper]](https://arxiv.org/abs/1811.03604)
- Rieke, N., et al. (2020). *The Future of Digital Health with Federated Learning*. npj Digital Medicine. [[Paper]](https://www.nature.com/articles/s41746-020-00323-1)
- Lim, W. Y. B., et al. (2020). *Federated Learning in Mobile Edge Networks: A Comprehensive Survey*. IEEE Communications Surveys & Tutorials. [[Paper]](https://arxiv.org/abs/1909.11875)

---

## 2. Core Challenges in Federated Learning

### 2.1 Statistical Heterogeneity (Non-IID Data)

In real-world FL, client data is rarely independent and identically distributed (IID). Data heterogeneity manifests as (Hsieh et al., 2020):

1. **Label distribution skew**: Clients have different label frequencies
2. **Feature distribution skew**: Same labels, different feature distributions
3. **Quantity skew**: Varying amounts of data per client
4. **Concept drift**: Data distributions change over time

**Impact**: Non-IID data causes *client drift*—local models diverge from the global optimum during local training, leading to slower convergence and lower accuracy (Zhao et al., 2018).

**Solutions**:
- Data augmentation and sharing strategies (Zhao et al., 2018)
- Proximal regularization (FedProx) (Li et al., 2020a)
- Variance reduction (SCAFFOLD) (Karimireddy et al., 2020)
- Personalization approaches (Fallah et al., 2020)

### 2.2 Systems Heterogeneity

Clients vary in:
- **Computational capacity**: Mobile phones vs. data centers
- **Network bandwidth**: 3G vs. fiber
- **Availability**: Intermittent connectivity, battery constraints

**Strategies**:
- Asynchronous aggregation (Xie et al., 2019)
- Client selection based on resources (Nishio & Yonetani, 2019)
- Partial participation (McMahan et al., 2017)

### 2.3 Communication Efficiency

Communication is often the bottleneck in FL (Konečný et al., 2016):
- Model updates can be megabytes to gigabytes
- Bandwidth is limited, especially for mobile devices
- Round-trip latency affects convergence time

**Compression Techniques**:
- **Gradient compression**: Quantization, sparsification (Alistarh et al., 2017)
- **Model compression**: Pruning, knowledge distillation (Caldas et al., 2018)
- **Local computation**: More local steps, fewer communication rounds (McMahan et al., 2017)

### 2.4 Privacy and Security

Even without sharing raw data, FL faces threats (Lyu et al., 2020):

1. **Inference attacks**: Reconstructing training data from gradients (Zhu et al., 2019)
2. **Membership inference**: Determining if a sample was in training set (Shokri et al., 2017)
3. **Model poisoning**: Malicious clients corrupting the global model (Bagdasaryan et al., 2020)
4. **Byzantine attacks**: Arbitrary malicious behavior (Blanchard et al., 2017)

**Defenses**:
- Differential privacy (Abadi et al., 2016)
- Secure aggregation (Bonawitz et al., 2017)
- Byzantine-robust aggregation (Yin et al., 2018)

### References

- Hsieh, K., Phanishayee, A., Mutlu, O., & Gibbons, P. (2020). *The Non-IID Data Quagmire of Decentralized Machine Learning*. ICML. [[Paper]](https://arxiv.org/abs/1910.00189)
- Zhao, Y., et al. (2018). *Federated Learning with Non-IID Data*. arXiv. [[Paper]](https://arxiv.org/abs/1806.00582)
- Li, T., et al. (2020a). *Federated Optimization in Heterogeneous Networks*. MLSys (FedProx). [[Paper]](https://arxiv.org/abs/1812.06127)
- Karimireddy, S. P., et al. (2020). *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning*. ICML. [[Paper]](https://arxiv.org/abs/1910.06378)
- Konečný, J., et al. (2016). *Federated Learning: Strategies for Improving Communication Efficiency*. NeurIPS Workshop. [[Paper]](https://arxiv.org/abs/1610.05492)
- Alistarh, D., et al. (2017). *QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding*. NeurIPS. [[Paper]](https://arxiv.org/abs/1610.02132)
- Zhu, L., Liu, Z., & Han, S. (2019). *Deep Leakage from Gradients*. NeurIPS. [[Paper]](https://arxiv.org/abs/1906.08935)
- Bonawitz, K., et al. (2017). *Practical Secure Aggregation for Privacy-Preserving Machine Learning*. CCS. [[Paper]](https://eprint.iacr.org/2017/281)

---

## 3. Federated Learning Algorithms

### 3.1 FedAvg: The Foundation

**Federated Averaging (FedAvg)** (McMahan et al., 2017) is the foundational FL algorithm:

```
Algorithm: FedAvg
─────────────────────────────────────────
For each round t = 1, 2, ..., T:
    1. Server selects subset S_t of K clients
    2. Server sends global model w_t to selected clients
    3. Each client k ∈ S_t:
       - Initialize local model: w_k = w_t
       - For E local epochs:
           - w_k ← w_k - η∇F_k(w_k; batch)
       - Send w_k to server
    4. Server aggregates: w_{t+1} = Σ_k (n_k/n) w_k
─────────────────────────────────────────
```

**Key insight**: Multiple local SGD steps before aggregation reduces communication by factor E while maintaining convergence.

### 3.2 FedProx: Handling Heterogeneity

**FedProx** (Li et al., 2020a) addresses statistical and systems heterogeneity by adding a proximal term:

$$\min_w F_k(w) + \frac{\mu}{2}\|w - w_t\|^2$$

The proximal term (controlled by μ) prevents local models from drifting too far from the global model during local training.

**When to use**: Non-IID data, clients with varying compute capabilities.

### 3.3 SCAFFOLD: Variance Reduction

**SCAFFOLD** (Karimireddy et al., 2020) uses *control variates* to correct for client drift:

$$w_k \leftarrow w_k - \eta(\nabla F_k(w_k) - c_k + c)$$

where:
- $c$ is the global control variate (estimates global gradient)
- $c_k$ is the local control variate (estimates local gradient)

**Benefit**: Provably faster convergence than FedAvg, especially with high heterogeneity.

### 3.4 FedOpt: Adaptive Server Optimization

**FedOpt** (Reddi et al., 2021) applies adaptive optimizers (Adam, Yogi) at the server:

```
Server update: w_{t+1} = w_t + ServerOpt(Δw_t)
```

Variants include **FedAdam**, **FedYogi**, and **FedAdagrad**.

### 3.5 Personalization Approaches

For heterogeneous clients, personalized FL learns client-specific models:

| Method | Approach | Reference |
|--------|----------|-----------|
| **Per-FedAvg** | MAML-style meta-learning | Fallah et al., 2020 |
| **FedPer** | Personal layers + shared backbone | Arivazhagan et al., 2019 |
| **pFedMe** | Moreau envelopes for personalization | T Dinh et al., 2020 |
| **FedBN** | Local batch normalization | Li et al., 2021 |
| **Ditto** | Multi-task learning formulation | Li et al., 2021b |

### 3.6 Algorithm Comparison

| Algorithm | Communication | Non-IID Robustness | Convergence | Complexity |
|-----------|--------------|-------------------|-------------|------------|
| FedAvg | Low (E local steps) | Moderate | O(1/√T) | Low |
| FedProx | Low | Good | O(1/√T) | Low |
| SCAFFOLD | Low | Excellent | O(1/T) | Medium |
| FedOpt | Low | Good | Adaptive | Medium |
| FedNova | Low | Excellent | O(1/√T) | Low |

### References

- McMahan, B., et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS. [[Paper]](https://arxiv.org/abs/1602.05629)
- Li, T., et al. (2020a). *Federated Optimization in Heterogeneous Networks* (FedProx). MLSys. [[Paper]](https://arxiv.org/abs/1812.06127)
- Karimireddy, S. P., et al. (2020). *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning*. ICML. [[Paper]](https://arxiv.org/abs/1910.06378)
- Reddi, S., et al. (2021). *Adaptive Federated Optimization*. ICLR. [[Paper]](https://arxiv.org/abs/2003.00295)
- Fallah, A., Mokhtari, A., & Ozdaglar, A. (2020). *Personalized Federated Learning with Theoretical Guarantees*. NeurIPS. [[Paper]](https://arxiv.org/abs/2002.07948)
- Wang, J., et al. (2020). *Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization* (FedNova). NeurIPS. [[Paper]](https://arxiv.org/abs/2007.07481)

---

## 4. Differential Privacy Fundamentals

### 4.1 Definition and Intuition

**Differential Privacy (DP)** (Dwork et al., 2006) provides a mathematical framework for quantifying privacy:

> A randomized mechanism M satisfies (ε, δ)-differential privacy if for all neighboring datasets D, D' (differing in one record) and all measurable sets S:
> 
> $$P[M(D) \in S] \leq e^\varepsilon \cdot P[M(D') \in S] + \delta$$

**Intuition**: An adversary cannot reliably determine whether any individual's data was included in the computation, regardless of auxiliary information.

**Parameters**:
- **ε (epsilon)**: Privacy budget. Lower = more private. Typical: 1-10
- **δ (delta)**: Probability of privacy failure. Should be < 1/n. Typical: 1e-5

### 4.2 Key Properties

1. **Post-processing immunity**: Any computation on DP output remains DP
2. **Composition**: Privacy degrades gracefully under multiple queries
3. **Group privacy**: Protection extends to groups (with scaled ε)

### 4.3 Basic Mechanisms

#### Laplace Mechanism (ε-DP)
For function f with sensitivity Δf:
$$M(D) = f(D) + \text{Lap}(\Delta f / \varepsilon)$$

#### Gaussian Mechanism ((ε,δ)-DP)
For function f with L2 sensitivity Δ_2 f:
$$M(D) = f(D) + \mathcal{N}(0, \sigma^2 I)$$
where $\sigma = \Delta_2 f \cdot \sqrt{2\ln(1.25/\delta)} / \varepsilon$

### 4.4 Sensitivity

**Global sensitivity** of function f:
$$\Delta f = \max_{D, D' \text{ neighbors}} \|f(D) - f(D')\|$$

For gradients in ML:
- **Gradient clipping** bounds sensitivity: $\Delta = C$ (clipping norm)
- Enables DP-SGD (Abadi et al., 2016)

### 4.5 Composition Theorems

| Composition Type | Result after k queries |
|-----------------|----------------------|
| **Basic** | (kε, kδ)-DP |
| **Advanced** | (ε√(2k·ln(1/δ')), kδ + δ')-DP |
| **Rényi (RDP)** | Tighter, converts at end |

### References

- Dwork, C., McSherry, F., Nissim, K., & Smith, A. (2006). *Calibrating Noise to Sensitivity in Private Data Analysis*. TCC. [[Paper]](https://link.springer.com/chapter/10.1007/11681878_14)
- Dwork, C., & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy*. Foundations and Trends. [[Book]](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
- Abadi, M., et al. (2016). *Deep Learning with Differential Privacy*. CCS. [[Paper]](https://arxiv.org/abs/1607.00133)
- Near, J., & Abuah, C. (2021). *Programming Differential Privacy*. [[Book]](https://programming-dp.com/)

---

## 5. Differential Privacy in Federated Learning

### 5.1 Why DP for FL?

Even without sharing raw data, FL leaks information through model updates:

1. **Gradient inversion attacks** can reconstruct training images from gradients (Zhu et al., 2019)
2. **Membership inference** can determine if a sample was used (Nasr et al., 2019)
3. **Property inference** can extract dataset properties (Melis et al., 2019)

DP provides provable protection against these attacks.

### 5.2 DP-SGD in FL

**Differentially Private Stochastic Gradient Descent (DP-SGD)** (Abadi et al., 2016) modifies SGD:

```
For each batch:
    1. Compute per-sample gradients: g_i = ∇L(x_i; w)
    2. Clip gradients: ĝ_i = g_i / max(1, ||g_i||/C)
    3. Aggregate: ḡ = (1/B) Σ ĝ_i
    4. Add noise: g̃ = ḡ + N(0, σ²C²/B² · I)
    5. Update: w ← w - η · g̃
```

**Key parameters**:
- **C**: Clipping norm (bounds sensitivity)
- **σ**: Noise multiplier (controls privacy)
- **B**: Batch size (affects privacy amplification)

### 5.3 Privacy Accounting

Modern DP-SGD uses **Rényi Differential Privacy (RDP)** for accounting:

1. Track RDP guarantee per step: $(\alpha, \varepsilon_\alpha)$-RDP
2. Compose across steps: RDP values add
3. Convert to (ε, δ)-DP at the end

**Subsampling amplification**: When batch is random subsample (rate q), privacy improves significantly.

### 5.4 User-Level vs. Example-Level DP

| Level | Protection Unit | Use Case |
|-------|-----------------|----------|
| **Example-level** | Individual data point | Standard DP-SGD |
| **User-level** | All data from one user | Cross-device FL |

User-level DP requires accounting for all examples from each user, typically requiring more noise.

### 5.5 Privacy-Utility Trade-offs

The fundamental trade-off in DP-FL:

$$\text{Lower } \varepsilon \Rightarrow \text{More noise} \Rightarrow \text{Lower accuracy}$$

Mitigation strategies:
- **More training data**: Privacy cost amortized
- **Model architecture**: Some architectures more robust to noise
- **Hyperparameter tuning**: Optimal clipping, learning rate
- **Public data**: Pre-training on public data reduces private training needs

### References

- Zhu, L., Liu, Z., & Han, S. (2019). *Deep Leakage from Gradients*. NeurIPS. [[Paper]](https://arxiv.org/abs/1906.08935)
- Nasr, M., Shokri, R., & Houmansadr, A. (2019). *Comprehensive Privacy Analysis of Deep Learning*. IEEE S&P. [[Paper]](https://arxiv.org/abs/1812.00910)
- Abadi, M., et al. (2016). *Deep Learning with Differential Privacy*. CCS. [[Paper]](https://arxiv.org/abs/1607.00133)
- McMahan, H. B., et al. (2018). *Learning Differentially Private Recurrent Language Models*. ICLR. [[Paper]](https://arxiv.org/abs/1710.06963)
- Geyer, R. C., Klein, T., & Nabi, M. (2017). *Differentially Private Federated Learning: A Client Level Perspective*. NeurIPS Workshop. [[Paper]](https://arxiv.org/abs/1712.07557)

---

## 6. DP Variants: Approximate, Rényi, Local, and Bayesian

### 6.1 Pure DP (ε-DP)

The original, strongest definition:

$$P[M(D) \in S] \leq e^\varepsilon \cdot P[M(D') \in S]$$

**Pros**: Clean semantics, simple composition
**Cons**: Too restrictive for deep learning (Gaussian noise doesn't satisfy pure DP)

### 6.2 Approximate DP ((ε,δ)-DP)

Relaxation allowing small failure probability δ:

$$P[M(D) \in S] \leq e^\varepsilon \cdot P[M(D') \in S] + \delta$$

**Interpretation**: With probability (1-δ), we have ε-DP; with probability δ, all bets are off.

**Key result**: Enables Gaussian mechanism, essential for DP-SGD.

**Advanced composition** (Dwork et al., 2010):
After k mechanisms, total privacy is approximately $(O(\varepsilon\sqrt{k}), k\delta)$-DP.

### 6.3 Rényi Differential Privacy (RDP)

**Definition** (Mironov, 2017): M satisfies (α, ε)-RDP if:

$$D_\alpha(M(D) \| M(D')) \leq \varepsilon$$

where $D_\alpha$ is the Rényi divergence of order α.

**Why RDP?**
1. **Tighter composition**: ε values add directly (for same α)
2. **Subsampling**: Clean amplification bounds
3. **Conversion**: Can convert to (ε, δ)-DP when needed

**Subsampled Gaussian Mechanism** (Mironov et al., 2019):
When sampling rate is q, RDP improves significantly—foundation of modern DP-SGD accounting.

### 6.4 Local Differential Privacy (LDP)

Each user perturbs their own data before sending:

$$P[M(x) = y] \leq e^\varepsilon \cdot P[M(x') = y]$$

**Key difference**: No trusted curator; privacy guaranteed even if aggregator is malicious.

**Trade-off**: Much more noise needed compared to central DP.

**Real-world deployments**:
- **Google RAPPOR**: Chrome telemetry (Erlingsson et al., 2014)
- **Apple**: iOS usage statistics (Apple, 2017)
- **Microsoft**: Windows telemetry

### 6.5 Concentrated DP (zCDP)

**Definition** (Bun & Steinke, 2016): M satisfies ρ-zCDP if for all α > 1:

$$D_\alpha(M(D) \| M(D')) \leq \rho \cdot \alpha$$

**Relation to other definitions**:
- ρ-zCDP implies (ρ + 2√(ρ·ln(1/δ)), δ)-DP
- Gaussian mechanism: N(0, σ²) on sensitivity-Δ query gives (Δ²/2σ²)-zCDP

### 6.6 Bayesian Differential Privacy

Standard DP is worst-case. Bayesian approaches consider:

1. **Prior on datasets**: What databases are likely?
2. **Posterior inference**: What can adversary learn?

**Approaches**:
- **Posterior sampling** (Dimitrakakis et al., 2017): Sample from posterior, analyze privacy
- **Data-dependent bounds** (Triastcyn & Faltings, 2019): Tighter guarantees for actual data
- **Bayesian accounting** (Zanella-Béguelin et al., 2022): Estimate ε from attack outcomes

### 6.7 Comparison Summary

| Variant | Composition | Trust Model | Tightness | Main Use |
|---------|-------------|-------------|-----------|----------|
| Pure ε-DP | Add ε | Central | Baseline | Theory |
| (ε,δ)-DP | Advanced | Central | Moderate | Standard ML |
| RDP | Add ε (per α) | Central | Tight | DP-SGD accounting |
| zCDP | Add ρ | Central | Tight | Gaussian mechanisms |
| LDP | Add ε | None needed | Loose | Telemetry |
| Bayesian | Varies | Central | Data-dependent | Adaptive |

### References

- Mironov, I. (2017). *Rényi Differential Privacy*. CSF. [[Paper]](https://arxiv.org/abs/1702.07476)
- Bun, M., & Steinke, T. (2016). *Concentrated Differential Privacy*. arXiv. [[Paper]](https://arxiv.org/abs/1605.02065)
- Erlingsson, Ú., et al. (2014). *RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response*. CCS. [[Paper]](https://arxiv.org/abs/1407.6981)
- Triastcyn, A., & Faltings, B. (2019). *Bayesian Differential Privacy for Machine Learning*. ICML. [[Paper]](https://arxiv.org/abs/1901.09697)
- Wang, Y., Balle, B., & Kasiviswanathan, S. P. (2019). *Subsampled Rényi Differential Privacy and Analytical Moments Accountant*. AISTATS. [[Paper]](https://arxiv.org/abs/1808.00087)
- Awan, J. (2025). *Lecture Notes on Differential Privacy*. [[Notes]](https://jordan-awan.com/teaching/)

---

## 7. Distributed Optimization Foundations

### 7.1 Connection to Federated Learning

FL algorithms are special cases of distributed optimization methods:

| FL Algorithm | Optimization Interpretation |
|--------------|---------------------------|
| FedAvg | Local SGD with periodic averaging |
| FedProx | Proximal gradient methods |
| SCAFFOLD | Variance-reduced stochastic methods |
| Decentralized FL | Gossip/consensus optimization |

### 7.2 Consensus Optimization

**Problem**: Multiple agents minimize $\sum_i f_i(x)$ while agreeing on solution.

**Approach** (Nedić & Ozdaglar, 2009):
```
Each agent i:
    1. Local gradient step: x_i ← x_i - η∇f_i(x_i)
    2. Consensus step: x_i ← Σ_j W_ij x_j
```

where W is a doubly stochastic mixing matrix.

**Connection to FL**: FedAvg is a special case where W corresponds to averaging through central server.

### 7.3 ADMM for Distributed Learning

**Alternating Direction Method of Multipliers (ADMM)** (Boyd et al., 2011) solves:

$$\min f(x) + g(z) \text{ s.t. } Ax + Bz = c$$

via alternating updates and dual ascent.

**FL formulation**:
$$\min \sum_k F_k(w_k) \text{ s.t. } w_k = z \text{ (consensus)}$$

Leads to algorithms like **FedADMM** (Zhang & Zhu, 2021).

### 7.4 Variance Reduction Methods

SCAFFOLD's control variates connect to variance reduction in stochastic optimization:

| Method | Idea | Reference |
|--------|------|-----------|
| **SVRG** | Periodic full gradient | Johnson & Zhang, 2013 |
| **SAGA** | Stored gradients | Defazio et al., 2014 |
| **SCAFFOLD** | Control variates in FL | Karimireddy et al., 2020 |

**Key insight**: Reduce gradient variance → faster convergence.

### 7.5 Convergence Analysis

For smooth, non-convex objectives with bounded heterogeneity:

| Algorithm | Rate | Key Assumption |
|-----------|------|----------------|
| FedAvg | O(1/√T) | Bounded gradient dissimilarity |
| FedProx | O(1/√T) | + proximal term |
| SCAFFOLD | O(1/T) | Control variate correction |

### References

- Boyd, S., et al. (2011). *Distributed Optimization and Statistical Learning via ADMM*. Foundations and Trends. [[Paper]](https://stanford.edu/~boyd/papers/admm_distr_stats.html)
- Nedić, A., & Ozdaglar, A. (2009). *Distributed Subgradient Methods for Multi-Agent Optimization*. IEEE TAC. [[Paper]](https://arxiv.org/abs/0803.1202)
- Bertsekas, D. P., & Tsitsiklis, J. N. (1989). *Parallel and Distributed Computation: Numerical Methods*. Athena Scientific. [[Book]](http://www.athenasc.com/ppbook.html)
- Johnson, R., & Zhang, T. (2013). *Accelerating Stochastic Gradient Descent using Predictive Variance Reduction*. NeurIPS. [[Paper]](https://papers.nips.cc/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html)

---

## 8. Systems and Infrastructure

### 8.1 FL Frameworks

| Framework | Organization | Strengths | Best For |
|-----------|--------------|-----------|----------|
| **Flower** | Flower Labs | Flexibility, research-friendly | Research, prototyping |
| **FedML** | FedML Inc. | Cloud integration, production | Deployment at scale |
| **TensorFlow Federated** | Google | TF ecosystem | Production TF systems |
| **PySyft** | OpenMined | Privacy focus, MPC | Privacy research |
| **FATE** | WeBank | Enterprise features | Cross-silo industry |

### 8.2 Flower vs FedML

| Aspect | Flower | FedML |
|--------|--------|-------|
| **Philosophy** | Lightweight, modular | Full-featured platform |
| **Learning curve** | Gentle | Steeper |
| **Deployment** | Manual or via flwr CLI | Built-in cloud support |
| **Simulation** | Built-in | Built-in |
| **Production** | Growing support | Strong (AWS, etc.) |
| **Customization** | Very flexible | More opinionated |

**Recommendation**:
- Use **Flower** for research, experimentation, learning
- Use **FedML** for production deployments, cloud-native systems

### 8.3 HPC Considerations

For large-scale FL on HPC clusters:

1. **Multi-GPU**: Parallelize local training
2. **Multi-node**: Distribute clients across nodes
3. **Communication**: MPI or gRPC for aggregation
4. **Storage**: Distributed file systems for checkpoints

**SLURM patterns**:
```bash
# Launch FL experiment on HPC
#SBATCH --nodes=4
#SBATCH --gpus-per-node=2

srun --ntasks-per-node=2 python run_fl.py --client-id $SLURM_PROCID
```

### 8.4 Scalability Patterns

| Pattern | Description | Trade-off |
|---------|-------------|-----------|
| **Synchronous** | All clients finish before aggregation | Simple but slow (stragglers) |
| **Asynchronous** | Aggregate as updates arrive | Fast but stale gradients |
| **Semi-sync** | Wait for K of N clients | Balanced |
| **Hierarchical** | Edge servers + central | Reduced central communication |

### References

- Beutel, D. J., et al. (2020). *Flower: A Friendly Federated Learning Framework*. arXiv. [[Paper]](https://arxiv.org/abs/2007.14390)
- He, C., et al. (2020). *FedML: A Research Library and Benchmark for Federated Machine Learning*. arXiv. [[Paper]](https://arxiv.org/abs/2007.13518)
- Bonawitz, K., et al. (2019). *Towards Federated Learning at Scale: A System Design*. MLSys. [[Paper]](https://arxiv.org/abs/1902.01046)

---

## 9. Advanced Topics

### 9.1 Secure Aggregation

**Goal**: Server learns only aggregate, not individual updates.

**Techniques**:
- **Secure Multi-Party Computation (MPC)**: Cryptographic protocols
- **Homomorphic Encryption**: Compute on encrypted updates
- **Secret Sharing**: Split updates among parties

**Practical system**: Bonawitz et al. (2017) designed scalable secure aggregation for mobile FL.

### 9.2 Byzantine-Robust FL

**Threat**: Malicious clients send arbitrary updates.

**Defenses**:
- **Coordinate-wise median** (Yin et al., 2018)
- **Trimmed mean** (Yin et al., 2018)
- **Krum** (Blanchard et al., 2017)
- **Bulyan** (El Mhamdi et al., 2018)

### 9.3 Federated Analytics

Beyond model training: compute statistics over decentralized data.

**Examples**:
- Heavy hitters (frequent items)
- Quantiles
- Histograms

**Approach**: Combine secure aggregation + DP for private analytics.

### 9.4 Vertical Federated Learning

**Setting**: Same samples, different features across parties.

**Challenge**: Need to align samples without revealing identities.

**Techniques**:
- Private set intersection
- Split learning

### 9.5 Federated Continual Learning

**Challenge**: Data distributions shift over time.

**Approaches**:
- Elastic weight consolidation (Kirkpatrick et al., 2017)
- Federated continual learning (Yoon et al., 2021)

### References

- Bonawitz, K., et al. (2017). *Practical Secure Aggregation for Privacy-Preserving Machine Learning*. CCS. [[Paper]](https://eprint.iacr.org/2017/281)
- Blanchard, P., et al. (2017). *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent*. NeurIPS. [[Paper]](https://arxiv.org/abs/1703.02757)
- Yin, D., et al. (2018). *Byzantine-Robust Distributed Learning*. ICML. [[Paper]](https://arxiv.org/abs/1803.05880)

---

## 10. Practical Implementation Guide

### 10.1 Getting Started

```python
# Install fl_research library
pip install -e .

# Quick start with SCAFFOLD
from fl_research.models import SimpleCNN
from fl_research.data import DirichletPartitioner
from fl_research.strategies import SCAFFOLDServer, StandaloneSCAFFOLDClient
from fl_research.privacy import PrivacyAccountant
from fl_research.utils import set_seed, MetricsTracker

set_seed(42)
```

### 10.2 Workflow

1. **Data preparation**: Partition dataset across clients
2. **Model selection**: Choose architecture (DP-compatible if using DP)
3. **Strategy selection**: FedAvg → FedProx → SCAFFOLD based on heterogeneity
4. **Privacy budget**: Set ε, δ based on requirements
5. **Training**: Run federated rounds
6. **Evaluation**: Track accuracy, loss, privacy spent

### 10.3 Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| BatchNorm with DP | Use GroupNorm or LayerNorm |
| Too much clipping | Tune C based on gradient norms |
| High ε | More data, public pretraining |
| Slow convergence | More local epochs, SCAFFOLD |
| Stragglers | Partial participation, async |

### 10.4 Debugging Tips

1. **Verify IID baseline**: Ensure algorithm works with IID data first
2. **Monitor gradient norms**: Check if clipping is too aggressive
3. **Track per-client metrics**: Identify problematic clients
4. **Ablate components**: Remove DP, aggregation, etc. to isolate issues

---

## 11. Master Reading List

### Foundational Papers

1. McMahan, B., et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. [[FedAvg]](https://arxiv.org/abs/1602.05629)
2. Dwork, C., & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy*. [[DP Book]](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
3. Abadi, M., et al. (2016). *Deep Learning with Differential Privacy*. [[DP-SGD]](https://arxiv.org/abs/1607.00133)

### Comprehensive Surveys

4. Kairouz, P., et al. (2021). *Advances and Open Problems in Federated Learning*. [[500+ pages]](https://arxiv.org/abs/1912.04977)
5. Li, Q., et al. (2020). *Federated Learning: Challenges, Methods, and Future Directions*. [[Survey]](https://arxiv.org/abs/1908.07873)
6. Yang, Q., et al. (2019). *Federated Machine Learning: Concept and Applications*. [[Taxonomy]](https://arxiv.org/abs/1902.04885)

### FL Algorithms

7. Li, T., et al. (2020). *Federated Optimization in Heterogeneous Networks*. [[FedProx]](https://arxiv.org/abs/1812.06127)
8. Karimireddy, S. P., et al. (2020). *SCAFFOLD: Stochastic Controlled Averaging*. [[SCAFFOLD]](https://arxiv.org/abs/1910.06378)
9. Reddi, S., et al. (2021). *Adaptive Federated Optimization*. [[FedOpt]](https://arxiv.org/abs/2003.00295)

### Differential Privacy

10. Mironov, I. (2017). *Rényi Differential Privacy*. [[RDP]](https://arxiv.org/abs/1702.07476)
11. Near, J., & Abuah, C. (2021). *Programming Differential Privacy*. [[Practical DP]](https://programming-dp.com/)
12. Dwork, C., et al. (2010). *Boosting and Differential Privacy*. [[Advanced Composition]](https://arxiv.org/abs/0903.4341)

### Distributed Optimization

13. Boyd, S., et al. (2011). *Distributed Optimization via ADMM*. [[ADMM]](https://stanford.edu/~boyd/papers/admm_distr_stats.html)
14. Bertsekas, D. P., & Tsitsiklis, J. N. (1989). *Parallel and Distributed Computation*. [[Classic Book]](http://www.athenasc.com/ppbook.html)

### Security & Attacks

15. Zhu, L., et al. (2019). *Deep Leakage from Gradients*. [[Gradient Attacks]](https://arxiv.org/abs/1906.08935)
16. Bonawitz, K., et al. (2017). *Practical Secure Aggregation*. [[SecAgg]](https://eprint.iacr.org/2017/281)

### Systems

17. Beutel, D. J., et al. (2020). *Flower: A Friendly Federated Learning Framework*. [[Flower]](https://arxiv.org/abs/2007.14390)
18. He, C., et al. (2020). *FedML: A Research Library and Benchmark*. [[FedML]](https://arxiv.org/abs/2007.13518)

### Curated Paper Collections (Highly Recommended!)

19. **FedML Awesome-Federated-Learning** - *The most comprehensive list* (500+ papers organized by research area):
    - https://github.com/FedML-AI/FedML/blob/master/research/Awesome-Federated-Learning.md
    - Topics: Distributed Optimization, Non-IID, Privacy, Attacks, Communication Efficiency
    - Papers from ICML, NeurIPS, ICLR, CVPR with researcher affiliations
    - Regularly updated with new publications

20. **Awesome Federated Learning**: https://github.com/weimingwill/awesome-federated-learning
21. **Awesome FL on Graphs**: https://github.com/huweibo/Awesome-Federated-Learning-on-Graph-and-Tabular-Data

### Online Resources

- **Flower Documentation**: https://flower.ai/docs/
- **Opacus Documentation**: https://opacus.ai/
- **FedML Documentation**: https://doc.fedml.ai/ (Note: Now part of TensorOpera AI platform)
- **Programming DP Book**: https://programming-dp.com/
- **Google FL Comics**: https://federated.withgoogle.com/
- **FL Field Guide**: https://arxiv.org/abs/2107.06917 (Optimization-focused)

---

## Appendix: Quick Reference

### A. Privacy Budget Guidelines

| Use Case | Recommended ε | Notes |
|----------|---------------|-------|
| Strong privacy | 0.1 - 1 | Significant accuracy loss |
| Moderate privacy | 1 - 5 | Reasonable trade-off |
| Weak privacy | 5 - 10 | Better accuracy |
| Research baseline | 10+ | Privacy is secondary |

### B. Non-IID Severity (Dirichlet α)

| α Value | Heterogeneity | Description |
|---------|---------------|-------------|
| 0.01 | Extreme | Each client has ~1 class |
| 0.1 | High | Each client has 1-2 classes |
| 0.5 | Moderate | Some class imbalance |
| 1.0 | Low | Roughly balanced |
| 100+ | ~IID | Nearly uniform |

### C. Algorithm Selection Flowchart

```
Start
  │
  ├─ Is data IID? ──Yes──► Use FedAvg
  │
  No
  │
  ├─ Is heterogeneity mild? ──Yes──► Use FedProx (μ=0.01-0.1)
  │
  No
  │
  └─► Use SCAFFOLD
```

---

*Last updated: December 2024*
*Repository: https://github.com/sdodlapati3/federated-learning*
