echo "Downloading pt-br datasets. This may take a while"
echo "Downloading Sid dataset:"
wget -c -q --show-progress -O ./sid.tar.gz https://www.dropbox.com/s/0wxlweatglrr7wl/sid.tar.gz?dl=0
echo "Downloading VoxForge dataset:"
wget -c -q --show-progress -O ./voxforge-ptbr.tar.gz https://www.dropbox.com/s/wrguetal6xmrgta/voxforge-ptbr.tar.gz?dl=0
echo "Downloading LapsBenchmark1.4 dataset:"
wget -c -q --show-progress -O ./lapsbm.tar.gz https://www.dropbox.com/s/8aqm9ktulmnry6d/lapsbm.tar.gz?dl=0

echo "Extracting Sid dataset..."
mkdir -p sid
cd sid; tar -xzf ../sid.tar.gz; cd ..

echo "Extracting VoxForge dataset..."
mkdir -p voxforge
cd voxforge; tar -xzf ../voxforge-ptbr.tar.gz; cd ..

echo "Extracting LapsBenchmark1.4 dataset..."
mkdir -p lapsbm
cd lapsbm; tar -xzf ../lapsbm.tar.gz; cd ..

echo "Finished."
