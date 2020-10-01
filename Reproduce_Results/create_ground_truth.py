from sys import path
path.append("../QIK_Web/util/")

from pycocotools.coco import COCO
import numpy as np
import pickle
import tensorflow as tf
import tensorflow_hub as hub
import argparse

# Contants
DATA_DIR = 'data'
DATA_TYPE = '2017'
ANN_FILE = '{}/instances_{}.json'.format(DATA_DIR,DATA_TYPE)
CAPTIONS_FILE = '{}/captions_{}.json'.format(DATA_DIR,DATA_TYPE)
SENTENCE_ENCODER_MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder/2"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess to extract text from a folder of csv files.')
    parser.add_argument('-data', default="data/15K_Dataset.pkl", metavar='data', help='Pickled file.', required=False)
    args = parser.parse_args()

    # Loading annotations and creating an Index
    coco = COCO(ANN_FILE)
    coco_caps = COCO(CAPTIONS_FILE)

    # Loading the subset of images.
    image_subset = pickle.load(open(args.data, "rb"))

    # Creating a dictionary of all images and captions.
    imgIds = coco.getImgIds();

    # Iterating over all the images for obtaining all captions.
    for image in imgIds:
        cap_lst = []
        # Fetching the captions for the images fetched
        img = coco.loadImgs(image)[0]

        # Skipping all the images, that are not a part of the dataset.
        if img['file_name'] not in image_subset:
            continue

        annIds = coco_caps.getAnnIds(imgIds=img['id']);
        anns = coco_caps.loadAnns(annIds)

        for ann in anns:
            cap_lst.append(ann['caption'])
            # captions_lst.append(ann['caption'])

        data_dict[img['file_name']] = cap_lst

    # Loading the sentence similarity model
    embed = hub.Module(SENTENCE_ENCODER_MODULE_URL)

    similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
    similarity_message_encodings = embed(similarity_input_placeholder)

    # Computing the similarity between captions.
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])

        message_embeddings = session.run(similarity_message_encodings,
                                         feed_dict={similarity_input_placeholder: captions_lst})
        corr = np.inner(message_embeddings, message_embeddings)
        session.close()

    # Loading the precomputed results.
    pre_computed_results = pickle.load(open(PRE_COMPUTED_RESULTS_PATH, "rb"))