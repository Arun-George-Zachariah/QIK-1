# QIK
**Q**uerying **I**mages Using Contextual **K**nowledge is a large-scale image retrieval system for complex everyday scenes. It combines advances in deep learning and natural language processing in a novel way to capture the relationships between objects in everyday scenes. It uses XML to represent tree representations of captions (e.g., parse tree) and stores and indexes them in an XML database. Given a query image, it constructs an XPath query based on the image's caption and identifies a set of candidate matches by querying the XML database. For ranking the candidates, it uses the tree edit distance between the parse tree (or dependency tree) of query caption and that of a candidate's caption. For more details, see the below publications.

## Publications
Arun Zachariah, Mohamed Gharibi, and Praveen Rao - **QIK: A System for Large-Scale Image Retrieval on Everyday Scenes With Common Objects.** In ACM International Conference on Multimedia Retrieval (ICMR 2020), pages 126-135, Dublin, Ireland. [[PDF]](https://drive.google.com/file/d/1hcqZRtRFHK2eC39D8p2j-pCbl2CgVsvl/view) [[DOI]](https://doi.org/10.1145/3372278.3390682)


## System Setup

Execute `scripts/deploy_scripts/init.sh`

```
. init.sh [--home Home] [--qik QIK_Home] [--core QIK_Core] [-h | --help]
```
Eg:
```
cd scripts/deploy_scripts && . init.sh --home /mydata --qik /mydata/QIK --core QIK_Core
```

## To start the web engine.
```
cd <QIK_HOME>/QIK_Web && python manage.py runserver <IP>:8000
```

The UI can be accessed at http://\<IP\>:8000/search/image_search

## To construct the index.
List of images can be added to `MetaDataGenerator/images.txt` or the directory containing the images can be added to `MetaDataGenerator/constants.py`

To start the indexing process:
```
cd MetaDataGenerator && python process_images.py
```

## Initial Evaluation Results
Initial evaluation results reported in the paper can be found [here](Documents/QIK_ICMR_Eval_Final.xlsx).

To reproduces the results, follow the steps in the [README](QIK_Evaluation/README.md).

The scripts were tested on Ubuntu 16.04. If you would need assistance in setting up the system or if you have any other concerns, feel free to email [Arun Zachariah](mailto:azachariah@mail.missouri.edu) and [Praveen Rao](mailto:praveen.rao@missouri.edu). You can also report any issues [here](https://github.com/MU-Data-Science/QIK/issues).

## To cite using BibTeX
If you use QIK's code for your research, please cite our publication as below:
```
@inproceedings{QIK_ICMR_2020,
author = {Zachariah, Arun and Gharibi, Mohamed and Rao, Praveen},
title = {QIK: A System for Large-Scale Image Retrieval on Everyday Scenes With Common Objects},
year = {2020},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
doi = {10.1145/3372278.3390682},
booktitle = {Proceedings of the 2020 International Conference on Multimedia Retrieval},
pages = {126–135},
numpages = {10},
location = {Dublin, Ireland},
series = {ICMR ’20}
}
```

## Contributors

***Faculty:*** Praveen Rao (PI)

***PhD Students:*** Arun Zachariah, Mohamed Gharibi

## Acknowledgments
This work was supported by the National Science Foundation under Grant No. 1747751. ([NSF IUCRC Center for Big Learning](http://nsfcbl.org))
