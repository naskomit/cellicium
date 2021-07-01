ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# versions of software
ARG fastqc_version="0.11.9"
ARG kallisto_version="0.46.2"
ARG samtools_version="1.11"
ARG trimgalore_version="0.6.6"
ARG bedtools_version="2.29.2"
ARG featurecounts_version="2.0.1"

# Installing programs as root

USER root

RUN apt-get update && \
  apt-get -y install curl zlib1g-dev libbz2-dev liblzma-dev \
  libncurses5-dev libncursesw5-dev libcurl4-openssl-dev libcairo2-dev libpango1.0-dev tcl-dev tk-dev

# Install FastQC
RUN curl -fsSL http://www.bioinformatics.babraham.ac.uk/projects/fastqc/fastqc_v${fastqc_version}.zip -o /opt/fastqc_v${fastqc_version}.zip && \
    unzip /opt/fastqc_v${fastqc_version}.zip -d /opt/ && \
    chmod 755 /opt/FastQC/fastqc && \
    ln -s /opt/FastQC/fastqc /usr/local/bin/fastqc && \
    rm /opt/fastqc_v${fastqc_version}.zip

# Install Kallisto
RUN curl -fsSL https://github.com/pachterlab/kallisto/releases/download/v${kallisto_version}/kallisto_linux-v${kallisto_version}.tar.gz -o /opt/kallisto_linux-v${kallisto_version}.tar.gz && \
    tar xvzf /opt/kallisto_linux-v${kallisto_version}.tar.gz -C /opt/ && \
    ln -s /opt/kallisto_linux-v${kallisto_version}/kallisto /usr/local/bin/kallisto && \
    rm /opt/kallisto_linux-v${kallisto_version}.tar.gz

# Install STAR
RUN git clone https://github.com/alexdobin/STAR.git /opt/STAR && \
    ln -s /opt/STAR/bin/Linux_x86_64/STAR /usr/local/bin/STAR && \
    ln -s /opt/STAR/bin/Linux_x86_64/STARlong /usr/local/bin/STARlong

# Install SAMTools
RUN curl -fsSL https://github.com/samtools/samtools/releases/download/${samtools_version}/samtools-${samtools_version}.tar.bz2 -o /opt/samtools-${samtools_version}.tar.bz2 && \
    tar xvjf /opt/samtools-${samtools_version}.tar.bz2 -C /opt/ && \
    cd /opt/samtools-${samtools_version} && \
    make && \
    make install && \
    rm /opt/samtools-${samtools_version}.tar.bz2

# Install featureCounts
RUN curl -fsSL http://downloads.sourceforge.net/project/subread/subread-${featurecounts_version}/subread-${featurecounts_version}-Linux-x86_64.tar.gz -o /opt/subread-${featurecounts_version}-Linux-x86_64.tar.gz && \
    tar xvf /opt/subread-${featurecounts_version}-Linux-x86_64.tar.gz -C /opt/ && \
    ln -s /opt/subread-${featurecounts_version}-Linux-x86_64/bin/featureCounts /usr/local/bin/featureCounts && \
    rm /opt/subread-${featurecounts_version}-Linux-x86_64.tar.gz

# Install TrimGalore
RUN mkdir /opt/TrimGalore && \
    curl -fsSL https://github.com/FelixKrueger/TrimGalore/archive/${trimgalore_version}.zip -o /opt/TrimGalore/trim_galore_v${trimgalore_version}.zip && \
    unzip /opt/TrimGalore/trim_galore_v${trimgalore_version}.zip -d /opt/TrimGalore && \
    ln -s /opt/TrimGalore/trim_galore /usr/local/bin/trim_galore && \
    rm /opt/TrimGalore/trim_galore_v${trimgalore_version}.zip

# Install bedtools2
RUN curl -fsSL https://github.com/arq5x/bedtools2/releases/download/v${bedtools_version}/bedtools-${bedtools_version}.tar.gz -o /opt/bedtools-${bedtools_version}.tar.gz && \
    tar xvzf /opt/bedtools-${bedtools_version}.tar.gz -C /opt/ && \
    cd /opt/bedtools2 && \
    make && \
    cd - && \
    cp /opt/bedtools2/bin/* /usr/local/bin && \
    rm /opt/bedtools-${bedtools_version}.tar.gz

# Switch back to the jovyan user
USER jovyan

# Jupyter utilities
RUN \
  pip install plotly==4.14.3 "ipywidgets>=7.5" && \
  pip install jupyterlab_widgets && \
  jupyter labextension install jupyterlab-plotly@4.14.3 && \
  pip install tqdm ipympl aquirdturtle_collapsible_headings \
    boto3 awscli==1.19.*


# Machine learning
RUN \
  pip install scikit-learn==0.24.* && \
  pip install umap-learn==0.5.* && \
  pip install tensorflow==2.5.* && \
  pip install pymc3==3.11.* && \
  pip install torch && \
  pip install torch-scatter && \
  pip install torch-sparse && \
  pip install torch-cluster && \
  pip install torch-spline-conv && \
  pip install torch-geometric

# Biology & single cell tools
RUN  pip install \
  pybind11 hnswlib python-igraph louvain \
  scanpy==1.7.* \
  scvelo==0.2.* \
  velocyto==0.17.* \
  DCA==0.3.* \
  leidenalg==0.8.* \
  pysam==0.16.* \
  gseapy==0.10.* \
  progeny-py==1.0.* \
  dorothea-py==1.0.* \
  cutadapt \
  magic-impute

# RUN \
#  pip install vivarium-core && \
#  pip install vivarium-cobra && \
#  pip install vivarium-bioscrape && \
#  pip install vivarium-multibody &&
#  pip install biocrnpyler[all] && \
#  pip install kipoi


# This must be the last set of commands (otherwise pip breaks)
RUN \
  pip uninstall -y numba && \
  pip install numba==0.52.0 --ignore-installed
