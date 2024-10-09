# **2Psinapod** - *extraction of auditory neural response features from two-photon recordings.*

Hey de Villers-Sidani lab descendant!  

This repository is intended to provide a full analysis pipeline and guide to extracting auditory neuronal feature information from raw two-photon recordings. 

## Analysis Notebooks:

Feature extraction and analysis code is contained in Jupyter notebooks.  Each notebook covers a particular neuronal property or group of properties, e.g. response bandwidth or best frequency.  

They are all designed to start from the same point, taking the formatted and normalized neuronal response dictionaries for a full response cohort (e.g. post-psilocybin administration).  They are set up to analyze all animals together but containing 
the ID's in dictionaries allows you to select particular animals, or run mixed effects models. 





## External Tools Employed: 
The full pipeline uses a number of tools that help us get from TIFF files to neuronal tuning properties. These are:

**Suite2P** - A powerful tool (Python) that aids in motion correction, ROI identification and calcium signal extraction from raw data.  You can either download and use the GUI, or directly clone from the Github if you want to make changes yourself. 

https://github.com/MouseLand/suite2p

**ROIMatchPub** - An opensource tool in MATLAB that significantly speeds up the matching of neuronal ROI's across recordings, allowing for longitudinal analysis. 

https://github.com/ransona/ROIMatchPub

**TDTBin2Mat** - MATLAB package to convert TDT sound presentation hardware file types to more workable format for Python. 

https://www.tdt.com/docs/sdk/offline-data-analysis/offline-data-matlab/





Authors:  Conor Lane, Veronica Tarka

Contact: conor.lane1995@gmail.com
