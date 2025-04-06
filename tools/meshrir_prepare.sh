directory="data/MeshRIR"
mkdir -p $directory
cd $directory
wget https://zenodo.org/records/10852693/files/S1-M3969_npy.zip
unzip S1-M3969_npy.zip
python ../../tools/meshrir_split.py --base_folder ./
rm S1-M3969_npy.zip
rm -rf S1-M3969_npy
rm -rf __MACOSX
