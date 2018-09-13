# Hawkes Process Modeling of Adverse Drug Reactions
---

## About

This repo contains a C++ implementation of our paper, "Hawkes Process Modeling of Adverse Drug Reactions with Longitudinal Observational Data" [[PDF]](http://people.csail.mit.edu/yujia/assets/pdf/hawkes_pdf.pdf).

> @inproceedings{hawkes,  
>   title={Hawkes Process Modeling of Adverse Drug Reactions with Longitudinal Observational Data},  
>   author={Bao, Yujia and Kuang, Zhaobin and  Peissig, Peggy and Page, David and Willett, Rebecca },  
>   booktitle={Machine Learning for Healthcare Conference},  
>   year={2017}
> }

---

## Data
Each row in the data file represent the trajectory of one patient, which is a sequence of ordered time-event pairs separated by whitespace:
> t_1 m_1 t_2 m_2 ... t_n m_n

where 0 <= t_1 <= ... <= t_n and m_i in {1,...,numOfVariables}. 

We assume {1,...,numOfOutcomes} to be the indices for the adverse outcomes and {numOfOutcomes+1, ..., numOfVariables} to be the indices for the drugs of interest.

The Marshfield Clinic EHR data is not publicly available due to patients' privacy. We provide a synthetic data generated from a Poisson autoregressive model for illustration. 

---

## Code Usage
1. Compile
```
g++ --std=c++14 -pthread Hawkes.cpp -o Hawkes.out
```

2. Example run
```
./Hawkes.out -f path_to_data -k num_of_kernels -w length_of_windows -l lasso_regularizations
```

---
### License

The MIT License (MIT)

Copyright (c) 2017 Yujia Bao 
