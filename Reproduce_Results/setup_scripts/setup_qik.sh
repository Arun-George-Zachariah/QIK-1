#!/usr/bin/env bash

# Constants.
HOME=${HOME}
QIK_HOME=${PWD}/../..
QIK_CORE_NAME=QIK
USER=${USER}

# Downloading Large Files.
wget https://mailmissouri-my.sharepoint.com/:u:/g/personal/az2z7_umsystem_edu/EQ8OVRhr8kdAoL-n52IX-sEBEsUhOrK5Q_C_SkqTUAthUQ?download=1 -O $QIK_HOME/ML_Models/ShowAndTell/checkpoints/model.ckpt-5000000.data-00000-of-00001
wget https://mailmissouri-my.sharepoint.com/:u:/g/personal/az2z7_umsystem_edu/EVA9jI60C4xAoaxgmnt4s2EB4evfyx4PFs6X6Z4lzkii2Q?download=1 -O $QIK_HOME/IndexEngine/lib/stanford-corenlp-3.9.2-models.jar
wget https://mailmissouri-my.sharepoint.com/:u:/g/personal/az2z7_umsystem_edu/ERWi9gvm7epHpogQIhr8d8EB0-krRTswMoo6LjKpgPuDQg?download=1 -O $QIK_HOME/IndexEngine/lib/stanford-parser-3.9.2-models.jar

# Changing the IP address from localhost.
bash $QIK_HOME/scripts/deploy_scripts/change_ip.sh

# Setting up BaseX
cd $QIK_HOME && wget http://files.basex.org/releases/9.2/BaseX92.zip && unzip BaseX92.zip && mv basex BaseX
bash $QIK_HOME/BaseX/bin/basexserver -S
bash $QIK_HOME/BaseX/bin/basex -c "create db QIK"

# Setting up Solr.
cd $HOME && wget https://archive.apache.org/dist/lucene/solr/8.0.0/solr-8.0.0.tgz
tar -xvf solr-8.0.0.tgz
chmod +x solr-8.0.0/bin/*
export SOLR_ULIMIT_CHECKS=false
echo 'export SOLR_ULIMIT_CHECKS=false' >> /users/$USER/.profile
. /users/$USER/.profile

# Starting Solr.
./solr-8.0.0/bin/solr start

# Creating QIK core.
./solr-8.0.0/bin/solr create -c $QIK_CORE_NAME

# Installing APTED
cd $QIK_HOME && git clone https://github.com/JoaoFelipe/apted.git APTED

# Building and deploying the indexing engine.
bash $QIK_HOME/scripts/deploy_scripts/deploy_index_engine.sh

# Directory to save the query image.
mkdir $HOME/apache-tomcat/webapps/QIK

# Setting Stanford Parser Classpath
export CLASSPATH=$QIK_HOME/IndexEngine/lib
echo 'export CLASSPATH=$QIK_HOME/IndexEngine/lib' >> /users/$USER/.profile

# Installing Python dependencies.
pip install -r $QIK_HOME/scripts/deploy_scripts/requirements.txt

# Clean up unwanted files.
rm -rvf $HOME/bazel-0.25.3-installer-linux-x86_64.sh
rm -rvf $HOME/jdk-8u131-linux-x64.tar.gz
rm -rvf $HOME/apache-tomcat-9.0.20.zip
rm -rvf $HOME/solr-8.0.0.tgz
rm -rvf $QIK_HOME/BaseX92.zip

# Executing profile.
echo '. '$HOME'/.bashrc' >> /users/$USER/.profile
