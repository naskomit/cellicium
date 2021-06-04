ARG BASE_IMAGE

FROM ${BASE_IMAGE}

RUN \
  pip install plotly==4.14.3 "ipywidgets>=7.5" && \
  pip install jupyterlab_widgets && \
  jupyter labextension install jupyterlab-plotly@4.14.3 && \
  pip install tqdm && \
  pip install biocrnpyler[all] && \
  pip install vivarium-core && \
  pip install vivarium-cobra && \
  pip install vivarium-bioscrape && \
  pip install vivarium-multibody

RUN \
  pip install tensorflow && \
  pip install umap-learn && \
  pip install -U scikit-learn && \
  pip install kipoi && \
  pip install pymc3
  
