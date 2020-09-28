# Getting Linux
FROM ubuntu:18.04

#  Creating the Working directory and copying the contents.
RUN mkdir /QIK_Web
WORKDIR /QIK_Web
COPY QIK_Web/ /QIK_Web
COPY ML_Models /ML_Models
COPY BaseX /BaseX
COPY IndexEngine /IndexEngine

ENV CLASSPATH /IndexEngine/lib

# Installing wget, unzip and ant.
RUN DEBIAN_FRONTEND=noninteractive \
	apt-get update \
	&& apt-get install -y wget \
	&& apt-get install -y unzip \
	&& apt-get install -y ant \
	&& apt-get install -y lsof \
        && apt-get install -y ant \
	&& apt-get -y install software-properties-common

# Installing Java.
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get -y install openjdk-8-jre-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# SOLR Set up.
RUN wget https://archive.apache.org/dist/lucene/solr/8.0.0/solr-8.0.0.tgz \
	&& tar -xvf solr-8.0.0.tgz \
	&& mv solr-8.0.0 solr \
	&& chmod +x /QIK_Web/solr/bin/*

# Intalling python dependencies.
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install -r /QIK_Web/requirements.txt

# Copying startup script into known file location in container.
COPY scripts/start.sh /start.sh

# Setting up Tomcat.
RUN wget https://archive.apache.org/dist/tomcat/tomcat-9/v9.0.20/bin/apache-tomcat-9.0.20.zip \
    && unzip apache-tomcat-9.0.20.zip \
    && mv apache-tomcat-9.0.20 apache-tomcat \
    && chmod +x /QIK_Web/apache-tomcat/bin/*

# Building the Index Enging.
RUN ant -buildfile /IndexEngine/build.xml

# Deploying the Index Engine.
RUN cp /IndexEngine/build/war/IndexEngine.war /QIK_Web/apache-tomcat/webapps

# Adding the QIK webapp.
RUN mkdir /QIK_Web/apache-tomcat/webapps/QIK

# Exposing the ports.
EXPOSE 8080 8000 8983

#CMD ["/start.sh"]