# Stochastic Optimization Project

This project studies stochastic optimization problems under uncertainty using scenario-based methods.
The focus is on approximation techniques and stability analysis for inventory and production planning problems.

The main problems considered are:
- Newsvendor problem
- Assemble-to-Order (ATO) problem

The project includes scenario reduction techniques based on moment matching and Wasserstein distance,
as well as in-sample and out-of-sample stability analysis.

---

## Project Structure

.
├── clustering/            # Scenario clustering and distance-based methods  
├── data/                  # Problem parameters  
├── problems/              # Problem definitions (Newsvendor, ATO)  
├── solvers/               # Stochastic optimization models  
├── stability/             # In-sample and out-of-sample stability analysis  
├── result/                # Result processing and storage  
├── setting/               # Scenario and experiment settings  
├── main.py                # Main experiments  
├── main_stability.py      # Stability analysis experiments  
├── requirements.txt  
└── README.md  

---

## Installation

To install the required dependencies, run the following command from the project root:
pip install -r requirements.txt

---

## Running the Code

To run the main experiments:
python main.py

To run the stability analysis:
python main_stability.py
