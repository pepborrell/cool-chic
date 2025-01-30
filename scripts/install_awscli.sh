# To make this run much faster, run this file from /scratch/jborrell
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install -i /scratch/jborrell/aws-cli -b /scratch/jborrell/bin

download_subset() {
	# Download one subset of images.
	if [ -e "/scratch/jborrell/openimages/$1" ]; then
	    echo "File already downloaded"
	    # We assume files were decompressed appropriately.
	else
	    /scratch/jborrell/bin/aws s3 --no-sign-request cp "s3://open-images-dataset/tar/$1" "/scratch/jborrell/openimages/$1"
       	    # Untar
	    tar -xzf "/scratch/jborrell/openimages/$1" -C /scratch/jborrell/openimages --checkpoint=100000 --checkpoint-action=echo="Processed %u %s records."
	fi
}

mkdir /scratch/jborrell/openimages
# Downloading several subsets until we get 0.5M images. Didn't get subset 1 because it was too large.
download_subset train_1.tar.gz
download_subset train_2.tar.gz
download_subset train_3.tar.gz
download_subset train_4.tar.gz
download_subset train_5.tar.gz
