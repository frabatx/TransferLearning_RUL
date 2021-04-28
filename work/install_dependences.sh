
echo "Update and install system dependences"
apt-get update --fix-missing
apt-get install ffmpeg libsm6 libxext6  -y
pip install --upgrade pip

echo "Update and install requirements"
pip3 install -r requirements.txt
