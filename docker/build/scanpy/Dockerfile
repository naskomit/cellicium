ARG BASE_IMAGE

FROM ${BASE_IMAGE}

# pip install ipympl
# jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib@0.7.4

RUN apt-get install -y bedtools && \
  pip install pybedtools && \
  pip install distfit && \
  Rscript -e "BiocManager::install(c('recount', 'DESeq2', 'derfinder', 'regionReport'), version = '3.12')"
