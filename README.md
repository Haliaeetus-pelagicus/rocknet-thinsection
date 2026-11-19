# RockNet round-8 checkpoint

Knowledge-constrained multi-task identification of lithology and minerals in petrographic thin sections.

This repository contains the best checkpoint of the eighth training stage and the inference protocol used for the reported hold-out numbers. It is prepared for a public GitHub archive that meets the Computers & Geosciences code requirements: an open-source license, an English user guide, pinned dependencies, and a runnable example. The repository is not distributed as one compressed archive.

## What is included

- `weights/model.pt.part00` and the following parts: the round-8 `model.pt` file, split so that every file is under GitHub's 100 MB limit. The joined file is the internal-validation checkpoint from epoch 2 of 12.
- `assets/label_maps.json`: 3 genetic groups, 108 formal lithology names, and 70 minerals. Names follow the Nanjing University teaching collection and are in Chinese.
- `assets/decode_tables.pt`: the hierarchy mask, mineral prior, and vernacular-name collapse used at decoding time.
- `examples/sample_tile.jpg`: one low-resolution tile for a smoke test. It is not a substitute for the hold-out set.

The 6168 hold-out tiles are in `data/holdout/`. They come from the public Nanjing University teaching collection (Lai et al., 2020; DOI `10.11922/sciencedb.j00001.00097`) and were held out by original photograph, not by tile.

## Computational requirements

- Python 3.10 or newer.
- About 2 GB of free RAM. A CUDA GPU is optional. Eight-view inference of one 384 x 384 tile takes a few seconds on a recent GPU and longer on CPU.
- Disk space about 150 MB for the checkpoint parts.

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch must match the machine. If `pip install -r requirements.txt` installs a CPU-only build, that build is sufficient for the example.

## User guide

### Input

One or more RGB photomicrographs (JPG, PNG, or TIFF). The published model was trained on 3 x 3 tiles of a microscope field, resized to 384 x 384. A whole field that is resized in one step is a valid input, but it is not the training geometry.

Tiles from the same field must share a file-name prefix ending in `_row_column`, with row and column in `{1, 2, 3}`. Example: `slide3-7_1_1.jpg` through `slide3-7_3_3.jpg`.

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--topk` | 5 | How many lithology names to return. |
| `--threshold` | 0.45 | Minerals below this probability are omitted. |
| `--tta-views` | 8 | Test-time views. `8` is the published setting. |
| `--pair-gap` | 0.32 | Rerank the top two names only when they are a known confusing pair and their probability gap is smaller than this value. |
| `--device` | auto | `cuda` or `cpu`. |
| `--output` | none | Also write the JSON report to this path. |

### Decoding order

The default command follows the round-8 evaluation order:

1. Average the eight flipped and rotated views.
2. Combine the genetic-group and lithology scores with the teaching-table hierarchy.
3. Reweight lithology with the mineral prior.
4. Rerank a confusing pair when the gap is below 0.32.
5. Fold a fine formal name into its vernacular class when the teaching slides never used the fine name on its own.
6. If at least five of the tiles from one field vote for the same name, assign that name to the field. Fewer than five votes leaves each tile unchanged.

### Output

A JSON object on standard output:

- `checkpoint_epoch`: `2` for this release.
- `tiles`: one object per input file, with `group`, `lithology` (highest probability first), `minerals`, and `area_fraction`.
- `photo_vote`: one object per microscope field. `applied` is true only when the five-vote rule fires.

Lithology and mineral names are the Chinese labels in `assets/label_maps.json`. `area_fraction` is the spatial average of the segmentation head. Class 0 is reported as `other/matrix`. It is a weak map, not an expert pixel annotation.

### Expected behaviour of the example

```bash
python -m rocknet examples/sample_tile.jpg --topk 3
```

The process exits with status 0 and prints JSON. `checkpoint_epoch` is 2, `tta_views` is 8, and `lithology` contains three names with probabilities that sum to less than or equal to 1. The bundled file is a thumbnail, so its top name is only a smoke test. It is not a reported accuracy.

Identifying nine tiles of one field:

```bash
python -m rocknet field_*_*.jpg
```

## Hold-out numbers produced with this protocol

On 6168 tiles withheld by original photograph, with eight-view averaging, pair reranking at 0.32, and the nine-tile vote:

| Quantity | Value |
| --- | --- |
| Genetic group | 99.64% |
| Lithology, top 1 | 99.30% |
| Lithology, top 3 | 99.35% |
| Sedimentary / igneous / metamorphic lithology | 99.47% / 99.18% / 99.29% |

The same 6168 tiles are included in this repository:

| Folder | Tiles |
| --- | --- |
| `data/holdout/sedimentary` | 1713 |
| `data/holdout/igneous` | 2187 |
| `data/holdout/metamorphic` | 2268 |

`data/holdout/manifest.csv` lists every file. No individual image is larger than 1 MB, so GitHub accepts the files directly. The whole set is about 315 MB. Together with the checkpoint parts, the repository is about 430 MB.

Run one group, or the whole hold-out, with the published protocol:

```bash
python -m rocknet data/holdout/sedimentary --output sedimentary.json
python -m rocknet data/holdout/igneous data/holdout/metamorphic data/holdout/sedimentary --output holdout.json
```

Passing a directory reads every image in that directory. Tiles whose names share the `_row_column` prefix are voted together. A full pass is slow on CPU because each tile uses eight views.

## Code availability statement

The round-8 checkpoint and the inference code are released under the MIT license at `https://github.com/<owner>/rocknet-thinsection`. Replace `<owner>` with the GitHub account before citing the repository. The checkpoint file is stored as parts `weights/model.pt.part00` onward because GitHub rejects a single file larger than 100 MB. `python -m rocknet` joins and checks those parts in memory.

## Uploading to GitHub

From this directory:

```bash
git init
git add .
git commit -m "Release the round-8 RockNet checkpoint and inference code."
git branch -M main
git remote add origin https://github.com/<owner>/rocknet-thinsection.git
git push -u origin main
```

Do not commit a joined `weights/model.pt`. That file is larger than 100 MB and GitHub will reject the push. It is listed in `.gitignore`.
