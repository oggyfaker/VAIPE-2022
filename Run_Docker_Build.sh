
NAME_DOCKER=vaipe2022
REPO_PATH=/home/fruit/Desktop/VAIPE-2022/


# 1.Install Docker-Nvidia2  (Or https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
# curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - distribution=ubuntu20.04


# curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
#     sudo tee /etc/apt/sources.list.d/nvidia-docker.list


# sudo apt-get update
# sudo apt-get install -y nvidia-docker2
# sudo pkill -SIGHUP dockerd


# 2. Build Docker 
docker build -f Dockerfile -t $NAME_DOCKER .



# 3. Run Docker 
docker run -v $REPO_PATH:/root/VAIPE \
            --runtime=nvidia \
            -ti $NAME_DOCKER bin/bash

# source activate VAIPE2022
# cd /root/VAIPE