# SpectraVision

## Abstract

Converting irregularly spaced mass spectrometry (MS/MS) data into computable representation is a foundational architectural choice that sets the upper bound for both computational efficiency and model accuracy. Existing MS/MS representations for machine-learning (ML) models can be divided into two parts: (1) fine-grained binned vectors that preserve spectra resolution but are sparse; (2) transformer-sequence encoders that can avoid sparse binning but must process variable-length inputs with continuous, shift-sensitive m/z coordinates. Critically, both strategies trade computational efficiency for representation quality, creating significant time and memory bottlenecks. In this paper, we present SpectraVision, which encodes MS/MS spectra into compact image-tensors, preserves high-resolution m/z information and supports efficient learning. Efficacy of these image-tensors to extract spatial features from the spectrum image is demonstrated with a CNN model and a Vision Transformer to validate the ability of image-tensors to preserve long-range spectral dependencies. Both CNN and ViT models then exhibit superior Top-k peptide-retrieval accuracy when compared with binned-Vector and transformer-sequence encoders. An added advantage of SpectraVision based models, necessary for training large models, is usage of 87% fewer parameters (~6.4M versus 51.8M) than binned-vector approach. Zero-shot evaluation on multiple species, non-tryptic peptides, and immunopeptides showcases the robustness of vision-based models to extend and improve ML models for MS-based proteomics.

## Set up with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and the project dependencies (Python 3.11+):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

## Peptide Retrieval Scoring (MGF and FASTA)

Search spectra against a peptide FASTA and save the highest scoring peptide per spectrum:

```bash
uv run python retrieval_search/run_search.py \
  --mgf retrieval_search/sample_data/sample_data.mgf \
  --fasta retrieval_search/sample_data/sample_fasta.fasta \
  --output-dir outputs/search \
  --ppm-tolerance 20 \
  --download_weight=true
```

MGF and FASTA inputs can be local paths or HTTPS URLs. Each MGF spectrum needs `PEPMASS=`, `CHARGE=`, and fragment peaks (`m/z intensity` lines); no peptide sequence is needed. Each FASTA record must contain one peptide; proteins are not digested. The search returns `matches.csv`. It uses `SpectraVision-3C-CNN` by default; choose another model with `--model`.

## MGF to H5 Processing

For training or labeled H5 inference, each MGF spectrum must have `SEQ=` (or `PEPTIDE=`), `PEPMASS=`, `CHARGE=`, and fragment peaks (`m/z intensity`). Convert `.mgf` or `.mgf.gz` with:

```bash
uv run python scripts/mgf_to_h5.py path/to/labeled.mgf path/to/spectra.h5
```

The H5 contains `peaks` (the 200 most intense peaks, sorted by m/z), `sequence`, `charge`, `precursor_mass` (m/z), `protein`, and `spec_id`. Supported modifications are carbamidomethylation, oxidation, carbamylation, deamidation, acetylation, and pyroglutamate.

## Inference

Evaluate a model with the following command:

```bash
./run_inference.sh SpectraVision-1C-CNN --download_weight=true --data path/to/data.h5
```

Choose `SpectraVision-3C-CNN` or `SpectraVision-PeakViT` for another model. Add `--data path/to/test.h5` to evaluate your labeled H5. The command ranks theoretical and decoy peptides within each batch, reports top 1 and top 3 accuracy, and writes predictions to `outputs/predictions_batch1024/`.

## Training on Your Data

Training H5 files need `peaks` (`[N, P, 2]` m/z and intensity) and `sequence` (`[N]` peptide strings). Preprocess your data from mgf to h5 as described above.

```bash
./run_train.sh SpectraVision-1C-CNN --data path/to/training.h5
```

`--data` can also point to a directory of H5 files. Choose either other model name to train it. Training saves `best.pt` and `last.pt` in the selected config's output directory. To validate during training, set `validation_data` in [`configs/run.json`](configs/run.json).

## Cite
The paper is under review. As soon as it is accepted, we will update this section.

## License

This model and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of this model and its derivatives, which include models trained on outputs from the model or datasets created from the model, is prohibited and requires prior approval. Downloading the model requires prior registration on Hugging Face and agreeing to the terms of use. By downloading this model, you agree not to distribute, publish or reproduce a copy of the model. If another user within your organization wishes to use the model, they must register as an individual user and agree to comply with the terms of use. Users may not attempt to re-identify the deidentified data used to develop the underlying model. If you are a commercial entity, please contact the corresponding author.

## Contact

For any additional questions or comments, contact Fahad Saeed (fsaeed@fiu.edu).
