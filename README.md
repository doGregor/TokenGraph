# TokenGraph

This repository contains all code and pre-processed data to rerun the TokenGraph classification experiments. The text
graphs can be downloaded from [Google Drive](https://drive.google.com/file/d/1X4K9SAGnrFA1liy2vOJlpYXpc2ZZz5LG/view?usp=sharing)
and the `graphs` folder with subfolders for each dataset should be placed in the
`TokenGraph` folder. It is also possible to put the raw data in the `TokenGraph` folder and rerun graph creation as
part of the `main.py` script. The code to run training and evaluation is also located in `main.py`. There is a CONFIG
at the top of the script that allows to set different parameters (e.g., dataset to use, hyperparameters).


Below, we provide results for different ablation studies that we performed to evaluate the robustness of our approach.
![ablation study results](appendix/ablation.png)