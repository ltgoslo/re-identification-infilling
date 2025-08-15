<h2 align="center"><b><h3>Re-identification of De-identified Documents with Autoregressive Infilling</h3></b></h2><br>


<p align="center">
  <b>Lucas Georges Gabriel Charpentier and Pierre Lison</b>
</p>

<p align="center">
  <i>
    University of Oslo, Norwegian Computing Center (NR)<br>
    Language Technology Group<br>
  </i>
</p>
<br>

<p align="center">
  <a href="https://aclanthology.org/2025.acl-long.60/"><b>Paper</b></a><br>
</p>

_______

<br>

<h3 align="center"><b>Abstract</b></h3><br>

Documents revealing sensitive information about individuals must typically be de-identified. This de-identification is often done by masking all mentions of personally identifiable information (PII), thereby making it more difficult to uncover the identity of the person(s) in question. To investigate the robustness of de-identification methods, we present a novel, RAG-inspired approach that attempts the reverse process of _re-identification_ based on a database of documents representing background knowledge. Given a text in which personal identifiers have been masked, the re-identification proceeds in two steps. A retriever first selects from the background knowledge passages deemed relevant for the re-identification. Those passages are then provided to an infilling model which seeks to infer the original content of each text span. This process is repeated until all masked spans are replaced. We evaluate the re-identification on three datasets (Wikipedia biographies, court rulings and clinical notes). Results show that (1) as many as 80% of de-identified text spans can be successfully recovered and (2) the re-identification accuracy increases along with the level of background knowledge.

_______

<br>

The official implemetation of the ACL 2025 paper: Re-identification of De-identified Documents with Autoregressive Infilling


_______

<br>

## Contents of the repository

- `./dataset_creation/`: Contains scripts to create the different datasets used for training and testing the model. This includes the scripts to create the datasets to train the reader, retriever, and ranker as well as, scripts to create the de-identified documents.
- `./ranking/`: Contains scripts to do the ranking (final (optional) step of the method) of lists of candidate names based on a document (de-identified, original, or re-identified). The `ranking.py` script does the general ranking (taking a dataset, model, etc.) while the other two scripts `clinical_ranking.py` and `tab_ranking.py` showcase the code used during the paper (same as the one found in `ranking.py` but not modular).
- `.\re-identify\`: Contains scripts to do re-identification of de-identified documents based on a background level and trained (or zero-shot) model. In this case we have a seperate script for each dataset re-identified. Those marked with `zero_shot` indicate the ones used with mistral, while the others correspond to those used with the trained GLM.
- `.\score_dense_retrieval\`: Contains scripts to score the performance of the dense retrieval, as done in the paper. It is again done for each dataset sperately.
- `.\sparse_retrieval\`: Contains scripts to create a sparse retriever and do sparse retrieval (or step 1 in our pipeline).
- `.\train_ranker\ `: Contains scripts to fine-tune a BERT-based candidate ranker model.
- `.\train_reader\ `: Contains scripts to fine-tune a GLM model (in this case GLM-RoBERTa).
- `.\train_retriever\ `: Contains scripts to fine-tune a COLBERT (BERT-based) retriever model.



> [!Warning]
> A lot of the code is as is, and might require some modifications to adapt to your needs (paths, wandb, models used, etc.)

> [!Note]
> Depending on interest, more general versions of re-identifcation could be released in the future