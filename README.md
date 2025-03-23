[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Guardians of Tomorrow: Leveraging Responsible AI for Early Detection and Response to Criminal Threats

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The software and data in this repository are a snapshot of the software and data
that were used in the research reported on in the paper 
[Guardians of Tomorrow: Leveraging Responsible AI for Early Detection and Response to Criminal Threats](https://doi.org/10.1287/ijoc.2023.0488) by X. Sun, Q. Wang, L. Qiu, and W. Xu.

## Cite

To cite this software, please cite the [paper](https://doi.org/10.1287/ijoc.2023.0488) and the software, using the following DOI.

https://doi.org/10.1287/ijoc.2023.0488.cd

Below is the BibTex for citing this snapshot of the repository.

```
@misc{ResponsibleAIForCrimeDetection,
  author =        {Xiaotong Sun and Qili Wang and Liangfei Qiu and Wei Xu},
  publisher =     {INFORMS Journal on Computing},
  title =         {Guardians of Tomorrow: Leveraging Responsible AI for Early Detection and Response to Criminal Threats},
  year =          {2025},
  doi =           {10.1287/ijoc.2023.0488.cd},
  url =           {https://github.com/INFORMSJoC/2023.0488},
  note =          {Available for download at https://github.com/INFORMSJoC/2023.0488},
}  
```

## Description

The goal of this repository is to demonstrate our proposed design framework for responsible AI systems that support proactive crime detection and real-time response.

It contains Python codes implementing (1) the data synthesizer used to generate privacy-preserving data and (2) the machine learning models used to achieve crime detection results.

A synthetic dataset is also provided to support replication of the crime detection system presented in our paper.

## Requirements

Require Python 3.6.9 or later version.

Install the dependencies using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Data Files

The original data for our paper was obtained from a government department and contains private information; therefore, it is confidential. 

To support replication, we provide a synthetic dataset in the file `/data/sample.csv`.

## Code Files

There are 2 source code files (in `/src`).

`Data_Synthesizer_for_Preserving_Privacy.py`: It includes the construction of the Greedy Bayes network model (Algorithm 1 in DataLearner), the computation of noisy conditional probabilities (Algorithm 2 in DataLearner), the implementation of the DataGenerator, and the analysis of information loss.

`Machine_Learning_Methods_for_Predictive_Modeling.py`: It includes the training and testing of the machine learning model, as well as the feature importance analysis.

## Result Files

There are four types of result files (in `/results`).

`/BNStructure`: This folder contains the structure of the learned Greedy Bayes network model.

`/AttributeUniqueValues`: This folder contains the unique values of each feature for data generation.

`/NoisyConditionalProbability`: This folder contains the learned noisy conditional probability distributions.

`/PrivatePreservingDataset`: This folder contains the generated synthetic dataset.

