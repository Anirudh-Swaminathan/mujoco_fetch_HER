# Unity Reacher
## ECE276C Final Project - Fall 2020 UCSD MS ECE

### Team Members

 - [Alexander Behnke](https://github.com/alexanderbehnke333) - abehnke@eng.ucsd.edu
 - [Anirudh Swaminathan](https://github.com/Anirudh-Swaminathan) - aswamina@eng.ucsd.edu
 - [Sai Akhil Suggu](https://github.com/saiakhil0034) - saiakhil0034@eng.ucsd.edu

### Description

This repository houses code for the final project of ECE 276C - Robot Reinforcement Learning taken at UCSD in Fall 2020.

### Base Code

The base implementation of DDPG and D4PG on Unity Reacher environment was provided in [this](https://github.com/TomLin/RLND-project) repository.
The Unity Reacher environment is provided by [this](https://github.com/Unity-Technologies/ml-agents) repository.
The specific environment that is in use in this project is the Unity [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

### Installation Instructions
 - Install dependencies on top of python3.6.8 (the version on DSMLP server)
 ```bash
 pip3 install -r requirements.txt
 ```
  - Follow env installation instructions as given in the <b>Getting Started</b> section in problem 2 [README](./rnld_repo/p2-continuous-control/README.md).
  Ensure that when on server, follow the instruction marked (for AWS) to download the env without visualization in [this](rnld_repo/p2-continuous-control) directory
