# Reproduce Results
Here, we compare image retrieval performance of QIK and its competitors in terms of mAP and retrieval time. We also describe the evaluation details and steps to reproduce the results. 

## Setup
To install pre-requisites:
```
./setup_prereq.sh
conda activate qik_env
```

To setup CBIR systems 
* QIK
./setup_qik.sh 

* CroW
./setup_crow.sh

* FR-CNN
./setup_frcnn.sh 

* DIR
./setup_dir.sh


Create Datasets
python create_dataset_pickle.py 
 
Create Ground truth
python create_ground_truth.py

To Do:
Add a script that performs a dry run, to ensure all the indexes are loaded.

## Results
* Table 1:  QIKc vs QIKo : two-object combinations (avg. mAP)
    <table>
        <tr>
            <td rowspan="2"></td>
            <td align="center" colspan="4">τ=0.6</td>
            <td align="center" colspan="4">τ=0.7</td>
        </tr>
        <tr>
            <td align="center">k=2</td>
            <td align="center">k=4</td>
            <td align="center">k=8</td>
            <td align="center">k=16</td>
            <td align="center">k=2</td>
            <td align="center">k=4</td>
            <td align="center">k=8</td>
            <td align="center">k=16</td>
        </tr>
        <tr>
            <td align="center">QIKc</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td align="center">QIKo (0.9)</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td align="center">QIKo (0.8)</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
    </table>

* Table 2:  QIKc vs QIKo : three-object combinations (avg. mAP)
    <table>
        <tr>
            <td rowspan="2"></td>
            <td align="center" colspan="4">τ=0.6</td>
            <td align="center" colspan="4">τ=0.7</td>
        </tr>
        <tr>
            <td align="center">k=2</td>
            <td align="center">k=4</td>
            <td align="center">k=8</td>
            <td align="center">k=16</td>
            <td align="center">k=2</td>
            <td align="center">k=4</td>
            <td align="center">k=8</td>
            <td align="center">k=16</td>
        </tr>
        <tr>
            <td align="center">QIKc</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td align="center">QIKo (0.9)</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td align="center">QIKo (0.8)</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
    </table>

* Table 3:  QIKc vs QIKo : three-object combinations (avg. mAP)
    <table>
        <tr>
            <td rowspan="2"></td>
            <td align="center" colspan="4">τ=0.6</td>
            <td align="center" colspan="4">τ=0.7</td>
        </tr>
        <tr>
            <td align="center">k=2</td>
            <td align="center">k=4</td>
            <td align="center">k=8</td>
            <td align="center">k=16</td>
            <td align="center">k=2</td>
            <td align="center">k=4</td>
            <td align="center">k=8</td>
            <td align="center">k=16</td>
        </tr>
        <tr>
            <td align="center">QIKc</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td align="center">QIKo (0.9)</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td align="center">QIKo (0.8)</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
    </table>

*  Table 4: Results for two-object combinations (avg. of mAP)
    <table>
        <tr>
            <td rowspan="2"></td>
            <td align="center" colspan="4">τ=0.6</td>
            <td align="center" colspan="4">τ=0.7</td>
        </tr>
        <tr>
            <td align="center">k=2</td>
            <td align="center">k=4</td>
            <td align="center">k=8</td>
            <td align="center">k=16</td>
            <td align="center">k=2</td>
            <td align="center">k=4</td>
            <td align="center">k=8</td>
            <td align="center">k=16</td>
        </tr>
        <tr>
            <td align="center">QIK</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td align="center">CroW</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td align="center">FR-CNN</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td align="center">DIR</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td align="center">DELF</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td align="center">LIRE</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
    </table>

*  Table 5: Results for three-object combinations (avg. of mAP)
    <table>
        <tr>
            <td rowspan="2"></td>
            <td align="center" colspan="4">τ=0.6</td>
            <td align="center" colspan="4">τ=0.7</td>
        </tr>
        <tr>
            <td align="center">k=2</td>
            <td align="center">k=4</td>
            <td align="center">k=8</td>
            <td align="center">k=16</td>
            <td align="center">k=2</td>
            <td align="center">k=4</td>
            <td align="center">k=8</td>
            <td align="center">k=16</td>
        </tr>
        <tr>
            <td align="center">QIK</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td align="center">CroW</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td align="center">FR-CNN</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td align="center">DIR</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td align="center">DELF</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td align="center">LIRE</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
    </table>

*  Table 6: Results for four-object combinations (avg. of mAP)
    <table>
        <tr>
            <td rowspan="2"></td>
            <td align="center" colspan="4">τ=0.6</td>
            <td align="center" colspan="4">τ=0.7</td>
        </tr>
        <tr>
            <td align="center">k=2</td>
            <td align="center">k=4</td>
            <td align="center">k=8</td>
            <td align="center">k=16</td>
            <td align="center">k=2</td>
            <td align="center">k=4</td>
            <td align="center">k=8</td>
            <td align="center">k=16</td>
        </tr>
        <tr>
            <td align="center">QIK</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td align="center">CroW</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td align="center">FR-CNN</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td align="center">DIR</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td align="center">DELF</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
        <tr>
            <td align="center">LIRE</td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
            <td align="center"></td>
        </tr>
    </table>




Consolidate Query Results
python create_results_pickle.py -qik <QIK_CAPTIONS_RESULTS> -qik_objects_8 <QIK_OBJECTS(0.9)_RESULTS> -qik_objects_9 <QIK_OBJECTS(0.8)_RESULTS> -frcnn <FR-CNN RESULTS> -dir <DIR_RESULTS> -delf <DELF_RESULTS> -lire <LIRE RESULTS> -crow <CroW_RESULTS> -out <CONSOLIDATED_QUERY_RESULTS>

Pre-constructed results are available at `pre_constructed_data`. To create a consolidated query results pickle:
```
python create_results_pickle.py -qik pre_constructed_data/QIK_Captions_Pre_Results_Dict.txt -qik_objects_8 pre_constructed_data/QIK_Objects_8_Pre_Results_Dict.txt -qik_objects_9 pre_constructed_data/QIK_Objects_9_Pre_Results_Dict.txt -frcnn pre_constructed_data/Deep_Vision_Pre_Results_Dict.txt -dir pre_constructed_data/DIR_Pre_Results_Dict.txt -delf pre_constructed_data/DELF_Pre_Results_Dict.txt -lire pre_constructed_data/LIRE_Pre_Results_Dict.txt -crow pre_constructed_data/Crow_Pre_Results_Dict.txt -out pre_constructed_data/15K_Results.pkl
```



To get mAP Results
```
python get_mAP.py -image_data <LIST_OF_IMAGES> -threshold <.70|.60> -pre_computed_results <CONSOLIDATED_QUERY_RESULTS> -ground_truth <GROUND_TRUTH> -categories <CATEGORY_COMBINATIONS> -outfile <RESULT_LOGS> 
```
* 
python get_mAP.py -image_data <LIST_OF_IMAGES> -threshold <.70|.60> -pre_computed_results <QUERY_RESULTS> -ground_truth <GROUND_TRUTH> -categories <CATEGORY_COMBINATIONS> -outfile <RESULT_LOGS> 
