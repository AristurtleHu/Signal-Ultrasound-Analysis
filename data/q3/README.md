# Question 3 dataset

## Sets and identifiers

| Set | Recordings | Metadata | Labels |
|---|---:|---|---|
| reference | 45 (15 per class) | reference/labels.csv | Provided |
| stage1 | 15 (5 per class), shuffled | stage1/index.csv | Withheld |
| stage2 | 47, shuffled | stage2/index.csv | Withheld |

The reference and Stage 1 sets contain different recordings. Use only the reference set to choose your features and decision rules. Apply the same method in both stages. Only reference labels are supplied. Submit predictions for both task stages without access to their labels; the TA will grade both using private ground truth. Students are not required to report task-stage accuracy or discuss errors against withheld labels.

Identifiers are REF001-REF045, S1_001-S1_015, and S2_001-S2_047. IDs identify files; do not use the numeric order as a classification feature. Stage 1 and Stage 2 IDs and file order are anonymized and shuffled. For both task stages, original filenames, acquisition order, selected-frame numbers, shuffle keys and true labels are not included.

Class encoding: 1 = acrylic; 2 = sandpaper; 3 = kitchen paper. No per-class counts for Stage 2 are supplied.

## Signal format

Each signal CSV contains exactly 896 data rows and the header:

```text
sample_index,normalized_amplitude
```

sample_index runs from 1 through 896. The time coordinate is `(sample_index - 1) / fs`, in seconds. The sampling rate is **fs = 26.5216e6 Hz** for all Q3 signals. These are real, single-beam recordings exported from the same acquisition setup; one selected frame represents each recording. Amplitudes have already been converted from the ADC scale to [-1, 1]. Do not repeat raw-ADC conversion or assume each waveform was scaled to its own peak.

Metadata tables use `sample_id` and `signal_file`; only the reference table includes `true_class`. Stage 1 and Stage 2 index tables contain no label column. A `signal_file` path is relative to the directory containing its metadata table. Machine-readable shared settings are in metadata.json.

## Loading example

From the notebook, ROOT points to the package root:

```python
import csv
import numpy as np

q3_root = ROOT / 'data' / 'q3'
with (q3_root / 'reference' / 'labels.csv').open(encoding='utf-8-sig') as f:
    reference_meta = list(csv.DictReader(f))
first = reference_meta[0]
x = np.loadtxt(q3_root / 'reference' / first['signal_file'],
               delimiter=',', skiprows=1)[:, 1]
fs_q3 = 26.5216e6
time_q3_us = np.arange(x.size) / fs_q3 * 1e6
```

## Prediction files

The two templates in the package root already contain every required sample_id:

- predictions_stage1.csv: 15 Stage 1 IDs.
- predictions.csv: 47 Stage 2 IDs.

Fill only predicted_class with 1, 2 or 3. Keep each sample_id exactly once and retain the header `sample_id,predicted_class`. Do not submit blank labels, probability values or extra columns. Submit both files. Definitions of optional features are in ../../FEATURE_GUIDE.md; choosing the features and classification rule is your task.
