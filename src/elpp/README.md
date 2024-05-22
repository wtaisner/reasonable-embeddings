The code for the paper "Is the proof length a good indicator of hardness for reason-able embeddings?" submitted to 
NeSy 2023.
Notebooks' descriptions:

* [prepare_kbs.ipynb]() Generate the synthetic KBs. You need to execute this one first, as the KBs are not committed 
  to the repository(Section 2.4)
* [exp1.ipynb]() The code for RQ1 (Section 3.1)
* [exp2.ipynb]() The code for RQ2 (Section 3.2)
* [exp3.ipynb]() The code for RQ3 (Section 3.3)
* [exp3-analyze.ipynb]() Summarizing the output of [exp3.ipynb]() (Figure 1)
* [stats.ipynb]() Computing statistics on the generated KBs (Table 3)
* [experiments.py]() A script conducting experiments 1 and 3 sequentially, including simple global configuration at the top, the possibility to run on GPU, and more comprehensive saving logic.
