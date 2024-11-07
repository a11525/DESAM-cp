# DESAM_cp
DESAM_cp (Diagnosis-Enhanced Sentence Attention Model for Clinical Prediction) is a multi-modal neural network developed to predict clinical outcomes for ICU patients. This model integrates patients' initial clinical notes and real-time monitoring data to forecast outcomes such as in-hospital mortality, ICU mortality, and length of stay. By employing sentence-level self-attention mechanisms and auxiliary loss predicting diagnostic information at discharge, DESAM_cp effectively captures diagnostic information from clinical records, providing robust predictions that support medical professionals in decision-making for diagnosis and treatment.

This model was introduced in the paper:
**[S. Lee et al., "Enhancing Clinical Outcome Predictions through Auxiliary Loss and Sentence-Level Self-Attention," 2023 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), Istanbul, Turkiye, 2023, pp. 1210-1217](https://ieeexplore.ieee.org/abstract/document/10385852/footnotes#footnotes)**.




## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/DESAM_cp.git
   cd DESAM_cp
   ```

2. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate py39
   ```
## Preprocessing
1. Download MIMIC-III dataset
2. Clone the repository "https://github.com/bioinformatics-ua/PatientTM" for icd-9 to CCS mapping
3. Clone the repository "https://github.com/MLforHealth/MIMIC_Extract"
3. Run the preprocessing code in the preprocess_notebook folder in order.
 - 0.1.icd_ccs_code_mapping - mimic3 icd9 code to ccs code 
 - 0.2.icu_chart_lab_extra
 - 0.3.mimic_extract - mimic_extract for this project
 - 1.preprocess_multi_visit_icu_patient
 - 2.make_note_to_feautre - Before proceeding with the "make bert emb shape" part, you must proceed with /src/make_emb.py.
 - 3.ccs_labeling

## Usage
To train the model, navigate to the src folder:
```bash
python main.py --task=train --save_name=desam_cp --learning_rate=0.0001 --patience=10 --epoch=1000 --save_model 
```

To test the model using the DESAM_cp checkpoint:
Test in checkpoint DESAM_cp
```bash
python main.py --task=test --save_name=DESAM_cp
```

## Citation
If you use DESAM_cp in your research or project, please cite our paper:

@INPROCEEDINGS{10385852,
  author={Lee, Sanghoon and Jang, Gwanghoon and Kim, Chanhwi and Park, Sejeong and Yoo, Kiwoong and Kim, Jihye and Kim, Sunkyu and Kang, Jaewoo},
  booktitle={2023 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)}, 
  title={Enhancing Clinical Outcome Predictions through Auxiliary Loss and Sentence-Level Self-Attention}, 
  year={2023},
  volume={},
  number={},
  pages={1210-1217},
  keywords={Codes;Hospitals;Neural networks;Decision making;Data visualization;Predictive models;Resource management;electronic health records;mortality prediction;length of stay prediction;clinical outcome prediction;Multi-modal neural network;self-attention mechanisms;diagnosis prediction;auxiliary loss},
  doi={10.1109/BIBM58861.2023.10385852}}


## Contact

for questions or issues, please contact tomatong94@gmail.com or a11525@korea.ac.kr