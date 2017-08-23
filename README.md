# Hawkes-ADR

A C++ implementation of our paper, "Hawkes Process Modeling of Adverse Drug Reactions with Longitudinal Observational Data"

> @inproceedings{hawkes,  
>   title={Hawkes Process Modeling of Adverse Drug Reactions with Longitudinal Observational Data},  
>   author={Bao, Yujia and Kuang, Zhaobin and  Peissig, Peggy and Page, David and Willett, Rebecca },  
>   booktitle={Machine Learning for Healthcare Conference},  
>   year={2017}
> }

## Data format
Each row in the data file represent the trajectory of one patient, which is a sequence of ordered time-event pairs separated by whitespace:
> t_1 m_1 t_2 m_2 ... t_n m_n

where $0\leq t_1\leq\cdots\leq t_n$ and $m_i\in\{1,\ldots,\text{numOfVariables}\}$. We assume $\{1,\ldots,\text{numOfOutcomes}\}$ are the indices for the adverse outcomes and $\{\text{numOfOutcomes}+1,\ldots,\text{numOfVariables}\}$ are the idices for the drugs of interest.

## 

## Todo:

