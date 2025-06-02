sudo apt update
sudo apt install build-essential
pip3 install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip3 install accelerate==1.6.0
pip3 install datasets==3.5.0
pip3 install transformers==4.50.3
pip3 install huggingface-hub
pip3 install gemlite==0.4.4
pip3 install hqq==0.2.5
pip3 install timm==1.0.15
# pip3 install triton==3.2.0