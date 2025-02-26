# FaithfulDefense

**FaithfulDefense** is a method designed to balance interpretability and privacy of model explanations. It provides **faithful** explanations from inherently interpretable models while minimizing the information leaked about the underlying model. The framework defends against model extraction attacks by controlling how much decision boundary information is exposed.

To learn more about our algorithm, read the research paper **Models That Are Interpretable But Not Transparent**. 

## Features

- Implements various explanation strategies, including:
  - **FaithfulDefense Explanations** (maximum_coverage_greedy and maximum_coverage_mip)
  - **Random Explanations (Baseline)** (random)
  - **Base Rule Explanations (Baseline)** (base)
  - **No Explanations (Baseline)** (none)
- Supports different **query strategies**, including:
  - **Random Sampling**
  - **Perturbation-based Sampling**
  - **Importance-Weighted Active Learning (IWAL)**

## Usage
To run the main experiments, execute `run_exp.sh`. This runs the `run_experiments.py` script, which:
- Loads datasets
- Initializes the AttackDefend model
- Iterates through different explanation and query strategies

To simulate an attacker training a surrogate model, run `run_surrogate.py` script. 
