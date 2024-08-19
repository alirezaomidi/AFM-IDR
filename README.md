# minD score for AlphaFold-Multimer

## Prerequisites

You need to have ColabFold experiments saved in zip files. Use `--zip` option while running ColabFold for this purpose.
You also need to save the outputs of the distogram head. Use `--save-all` option while running ColabFold for this purpose.

# Calculating minD

Compute minD scores for the AlphaFold-Multimer models using the `compute_mind.py` script. The script will compute the minD score for each model and save the results in a json file within the experiment directory:

```bash
./armidr/compute_mind.py /path/to/colabfold/experiment --n-jobs 5
```
