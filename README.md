# EE180 Fall Homework 1 - student package

**Due: October 8, 2026, 23:59.** Read EE180_HW1.pdf first; it is the authoritative assignment.

## Files

- EE180_HW1.pdf: questions and submission requirements.
- EE180_HW1.tex: self-contained LaTeX source for the assignment PDF. Compile with XeLaTeX (run `xelatex EE180_HW1.tex` twice); no additional question source files are required.
- codes/HW1.ipynb: optional Python working template with data-loading examples.
- codes/feature_helpers.py and FEATURE_GUIDE.md: optional single-feature helpers and definitions.
- data/q1_signals.npz / q1_signals.mat: identical data for both Q1 and Q2, sampled at 60 MHz.
- data/q3/: labeled reference (45), and anonymized, shuffled Stage 1 (15) and Stage 2 (47) recordings with labels withheld, sampled at 26.5216 MHz; see data/q3/README.md.
- predictions_stage1.csv and predictions.csv: prefilled IDs and empty prediction columns.
- requirements.txt: Python dependencies.

## Run the notebook

Create or activate a Python environment, then run these commands from the extracted package root:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab codes/HW1.ipynb
```

Select the same environment as the notebook kernel. Starting Jupyter from the codes/ directory also works. Run cells in order. The setup cell creates image/ for your figures. Processing, classification and plotting cells marked TODO are intentionally left for you to complete. All Q3 recordings are now supplied; no external acquisition folder is needed. MATLAB users can use the MAT file and Q3 CSVs and implement their own analysis.

Q1 and Q2 use the same supplied signal data; there is no separate Q2 data file. Do not substitute the Q3 sampling rate for the Q1/Q2 sampling rate. No per-recording raw-ADC normalization is needed for Q3.

## Submit

Submit EE180-ID-Name-hw1.zip containing codes/, image/, report-ID-Name-hw1.pdf, predictions_stage1.csv and predictions.csv. Write your report in English, include brief running instructions, and use relative paths. The notebook does not replace the required PDF report. Fill all 15 Stage 1 predictions and all 47 Stage 2 predictions, each with a label 1, 2 or 3.

Only the reference-set labels are public. The TA grades both task stages using private labels. This package contains no reference classification thresholds, reference predictions or answer keys for either task stage. The optional feature guide is not a classification solution.
