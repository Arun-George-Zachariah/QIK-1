from sys import path
path.append("../QIK_Web/util/")

import constants
from qik_search import qik_search
import datetime

# Contants
TWO_CATEGORY_COMBINATIONS = "data/2_cat_comb.txt"
THREE_CATEGORY_COMBINATIONS = "data/3_cat_comb.txt"
FOUR_CATEGORY_COMBINATIONS = "data/4_cat_comb.txt"
EVAL_K = 16

def get_qik_results(query_image):
    ret_dict = {}

    # Reading the input request.
    query_image_path = constants.TOMCAT_LOC + constants.IMAGE_DATA_DIR + query_image

    # Get QIK results
    time = datetime.datetime.now()

    qik_pre_results = []
    qik_results = []

    # Fetching the candidates from QIK.
    qik_results_dict = qik_search(query_image_path, obj_det_enabled=False, ranking_func='Parse Tree', fetch_count=EVAL_K + 1)
    for result in qik_results_dict:
        k, v = result
        qik_pre_results.append(k.split("::")[0].split("/")[-1])

    # Noting QIK time.
    qik_time = datetime.datetime.now() - time
    print("qik_pre_eval :: retrieve:: QIK Fetch Execution time :: ", qik_time)

    # Removing query image from the result set.
    for res in qik_pre_results:
        if res == query_image:
            continue
        qik_results.append(res)

    # Adding data to the return dictionary.
    ret_dict["qik_time"] = qik_time.microseconds
    ret_dict["qik_results"] = qik_results

    print("qik_pre_eval :: retrieve :: ret_dict :: ", str(ret_dict))
    return ret_dict

if __name__ == "__main__":
    with open(TWO_CATEGORY_COMBINATIONS, 'r') as two_cat_comb:
        for combination in two_cat_comb.readlines():
            print(combination)