## CSCI 2470 Final Project

Title: From CLIP Features to Concepts for Interpretable Dog Breed Classification

### Repo Structure

- `cbm.ipynb`  
  The main notebook of this project. It contains the data preparation, core implementation, training and evaluation of the CBM models.

- `generate_concepts.py`  
  Code for generating concept-related data using LLMs. This script is used to produce intermediate files required by the main notebook.

- `concept_matrix.json`  
  Generated concept matrix used by the CBM.

- `llm_concepts.txt`, `label_free_concepts.txt`, `dog_breed_names.txt`  
  Text files generated during the concept generation stage and used as inputs to the model.