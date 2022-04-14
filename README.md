# NCI-DOE-Collab-Pilot3-RNN-LSTM-based-Clinical-Text-Generator

### Description
Given a sample corpus of biomedical text (i.e. pathology reports), this resource builds a long short-term memory (LSTM) model, a type of recurrent neural network (RNN), to automatically generate synthetic biomedical text of desired clinical context. This resource addresses the challenge of collecting labeled data, specifically clinical data, needed to create robust machine learning and deep learning models.

### User Community
Data scientists interested in generating more examples of unstructured text with a specific label from a given corpus. The data produced can be used for training machine learning or deep learning models on clinical text.

### Usability	
Data scientists can train the provided untrained model on their own data or with preprocessed data of clinical pathology reports from [SEER](https://seer.cancer.gov/) included with this resource. 

To use this resource, users must be familiar with natural language processing (NLP) and training neural networks, specifically RNNs.

### Uniqueness	
Generative models of text is a known problem in the natural language processing community. The model architecture provided in this resource was tested on unstructured clinical data and can be further optimized using the [CANDLE platform](https://datascience.cancer.gov/collaborations/joint-design-advanced-computing/candle). 

### Components	
* Data:
  * The preprocessed training and test data of SEER clinical pathology reports are in [MoDaC](https://modac.cancer.gov/assetDetails?returnToSearch=true&&dme_data_id=NCI-DME-MS01-18031472).

### Technical Details
Refer to this [README](./Pilot3/P3B2/README.md).

