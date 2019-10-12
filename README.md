# Machine-Learning  

IDEA: to predict if two drugs can be combined for various diseases  
DATA from:  15.2018-bioinformatics-stanford-Decagon &  8.2019-NC-Network-based prediction of drug combinations  
METHODS from: 15.2018-bioinformatics-stanford-Decagon or other embedding methods  

STEPS:   

1: data preprocess for knowledge graph, to integtate as much information as possible  (Oct 15)

   1, knowledge graph for the project is in /ProjectData  
   2, drug combinations is from /DrugCombination2019  
   3, basic knowledge graph is from /DTI_rimodel_kg_data 
   4, additional drug-disease, protein-disease and protein-protein interaction are from /DTI2017   
   

2: knowledge graph embedding for features  (Oct 30)  
   network embedding here: https://towardsdatascience.com/graph-embeddings-the-summary-cc6075aba007   
   knowledge graphs have different kinds of edges, the embedding is more difficult:          https://ieeexplore.ieee.org/iel7/69/4358933/08047276.pdf   

3: machine learning methods on the features to predict drug-drug combination  (Nov 20)  

4: interpretation/verification of the results, paper review, poster   (Nov 30)  
