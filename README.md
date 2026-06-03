# Drug-combination-identification-based-on-Knowledge-Graph

# Description
The process consists of five main steps: 1. Collect drug-drug combination information and drug-target interaction information from various databases, such as SynDrugMD, as well as from the literature. 2. Further organize the dataset and remove redundant information based on the collected drug CIDs, drug database names, and target IDs. 3. Based on the collected information, construct a knowledge graph and convert the chemical structure (SMILES) information of drugs and the sequence information of target proteins. 4. Utilize the pre-trained models ChemBert and Prot_Bert to extract feature vectors from drug chemical structure information and corresponding target sequence information, respectively, while employing the TransE model for embedding learning to represent drug-drug combinations. 5. Build an XGBoost model to identify potential drug-drug combinations.

# Dependencies
The source code developed in Python 3.8.20. The required python dependencies are given below.
1. pykeen=1.10.2
2. xgboost=2.1.4
3. scikit-learn=1.3.2
4. numpy=1.24.3
5. scipy=1.10.1
6. pandas=2.0.3

# Methods
1. Run feature.py
2. Run model.py
