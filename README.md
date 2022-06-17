# XAVI-AI4AD

This repository contains code implementing the method called **eXplainable Autonomous Vehicle Intelligence (XAVI)** described in the workshop paper:

["A Human-Centric Method for Generating Causal Explanations in Natural Language for Autonomous Vehicle Motion Planning"]()
by Gyevnar et al. [1], published at IJCAI Workshop on Artificial Intelligence for Autonomous Driving ([AI4AD](https://learn-to-race.org/workshop-ai4ad-ijcai2022/)), 2022.

XAVI builds on an existing inherently interpretable motion planning and prediction system called [IGP2](https://ieeexplore.ieee.org/abstract/document/9560849).
The open-source implementation of IGP2 can be found under the repository [uoe-agents/IGP2](https://github.com/uoe-agents/IGP2).

## Please cite

If you use the code in this repository in your work please cite "A Human-Centric Method for Generating Causal Explanations in Natural Language for Autonomous Vehicle Motion Planning".
You can use the bibtex template below in your document:

```latex
@inproceedings{gyevnar2022humancentric,
   title={A Human-Centric Method for Generating Causal Explanations in Natural Language for Autonomous Vehicle Motion Planning},
   author={Balint Gyevnar and Massimiliano Tamborski and Cheng Wang and Chrisopher G. Lucas and Shay B. Cohen and Stefano V. Albrecht},
   booktitle={IJCAI Workshop on Artificial Intelligence for Autonomous Driving (AI4AD)},
   year={2022}
```

## Installation and usage

XAVI uses IGP2, which relies on the powerful driving simulation software called [CARLA](https://carla.org/).
To install IGP2 and its dependencies, please refer to the README on the [IGP2 repo](https://github.com/uoe-agents/IGP2). 

All other dependencies will be installed from the requirements.txt file and no other external dependencies are required.

To install XAVI please clone and install the code in this repo using the following commands:

```bash
git clone https://github.com/uoe-agents/xavi-ai4ad.git
cd xavi-ai4ad
pip install -e .
```

To run the code and generate sample explanations for scenario 1 and 2 from IGP2, please run the following commands from the root directory of the repo:

```python
python scenarios/scripts/scenario1.py
python scenarios/scripts/scenario2.py
```

The output of the method will be logged to the console.

## References

[1] Balint Gyevnar, Massimiliano Tamborski, Cheng Wang, Chrisopher G. Lucas, Shay B. Cohen, Stefano V. Albrecht. A Human-Centric Method for Generating Causal Explanations in Natural Language for Autonomous Vehicle Motion Planning. IJCAI Workshop on Artificial Intelligence for Autonomous Driving (AI4AD), 2022.  