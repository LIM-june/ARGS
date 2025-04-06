# This folder contains tools to create the dataset

## Instructions on creating Mesh-RIR S1-M3969 split

1. First download the [MeshRIR S1-3969 dataset](https://www.sh01.org/MeshRIR/) in an empty folder

2. Run the following command
```sh
python meshrir_split.py --base_folder /path/to/the/empty/folder
```

3. Done! You have created your own Mesh-RIR dataset. The dataset structure should be similar to the following:

```sh
MeshRIR-Dataset/
├── S1-M3969_npy/          # Raw dataset
├── train/                 # train split
├── test/                  # test split
├── pos_mic.npy            # dataset metadata for microphone position
└── pos_src.npy            # dataset metadata for speaker position
```

⭐ or just run from the project base folder:
```sh
bash tools/meshrir_prepare.sh
```