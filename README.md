# Wikipedia-Open-QnA

Implementation of the Papers DRQA (Module 1 and Module 2) "Reading Wikipedia to Answer Open-Domain Questions" and Rank QA (Module 3) "RankQA: Neural Question Answering with Answer Re-Ranking"

## Team Members

- T H Arjun , 2019111012
- Arvindh A , 2019111010

## Data

As a part of the project we indexed the whole wikipedia, implemented the DRQA Retriever, DRQA Reader and extracted features required to run the RankQA Model as described in the paper.We then All the files required to run the modules and train and the reproduce the results are in the [folder](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/arjun_thekoot_research_iiit_ac_in/EiuPm6Hem95Aib5XDbNh05wB9XeMBeOBKUzecYuQ3IpWtw?e=XddpwF)

## How to run

- Install the required libraries
- The paths in files are absolute to the machine where we ran the experiments on, to rerun the experiments you will need to modify the path.
- All the contents of the uploaded directory are necessary to rerun the experiments.
- We have extracted all the features required for training which was computationally heavy and hence we have uploaded them to a shared folder as well.
- We have also given shell scripts required to extract these from the files if needed.

## Directory Structure

- `data` folder contains the SQuAD Dataset you might need to move things from data folder in the link to `data` folder and change the paths accordingly.
- `module_1` directory has the implementation of the DrQA retriever.
  - `download.sh` will download the wikipedia dump required to run the experiments.
  - `make_db.sh` will invoke the necessary python files to create the SQL database of the Wikipedia Dump.
  - `make_index.py` will create a searchable index of the Wikipedia dump with TF-IDF Vectorisation.
  - `make_index.sh` will invoke the necessary files files to create the searchable index.
  - `make_tfidf.py` will TF-IDF Vcetorise the Wikipedia Dump using multiprocessing.
  - `retriever.py` has the implementation of the DrQA retriever class which works with the files generated by the above files.
  - `run_squad.py` files can be used to generate the features for RankQA for SQuAD Dataset.
  - `retriever_example.ipynb` shows an example on how to use the retriever.
- `module_2` folder contains the implementation of DrQA Reader as told in the paper.

  - `models` folder should be created to put the checkpoint in. The checkpoint from our experiments are uploaded on the shared folder.
  - The various python files include the various layers, dataasets and other utilities required to train the pytorch model that we have implemented.
  - `drqa.py` has the DrQA class related to the model with `layers.py` , `dictionary.py` , `utils.py` have utilies required to train them.
  - `train.py` can be invoked by `train.sh` and necessary path changes to train the DrQA Reader from scratch.
  - `extract.ipynb` can be used along with `make_df.ipynb` for extracting features required for RankQA.
  - `plot.ipynb` plots some plots related to training from the report.
  - `test.ipynb` shows how to exactly use DrQA Reader for predictions.

- `module_3` has the implementation of RankQA model.

  - `images` folder has images from the report.
  - `convert.ipynb` converts the CSV files from previous steps to JSONL format for easy processing by RankQA.
  - `RankQA.ipynb` has all the implementation of RankQA, the notebook can be run after getting all the necessary files from the shared folder and after changing the absolute paths.

- `paper.pdf` contains the given Paper as a PDF.
- `Report.pdf` has the project report

## References

- Kratzwald, B., Eigenmann, A., & Feuerriegel, S. (2019). Rankqa: Neural question answering with answer re-ranking. arXiv preprint arXiv:1906.03008.
- Chen, D., Fisch, A., Weston, J., & Bordes, A. (2017). Reading Wikipedia to answer open-domain questions. arXiv preprint arXiv:1704.00051.
- Burges, Chris, et al. "Learning to rank using gradient descent." Proceedings of the 22nd international conference on Machine learning. ACM, 2005
