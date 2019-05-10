import sys
import os
import cv2
import numpy as np
from .facenet import prewhiten, load_model
import tensorflow as tf
from scipy import misc
from .matching_utils import *

EMBEDDING_SIZE = 160
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'matching/face_embedding/20180402-114759.pb', 'stored facenet-recognition model')
flags.DEFINE_string('f', '', 'kernel')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    return face_distance(known_face_encodings, face_encoding_to_check)

def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)

def encoding(fname, sess, images_placeholder, phase_train_placeholder, embeddings):
    face = cv2.imread(fname)
    face = misc.imresize(face, (EMBEDDING_SIZE, EMBEDDING_SIZE), interp="bilinear")
    face_prewhiten = prewhiten(face)
    face_encoded = facenet_embedding([face_prewhiten], sess, images_placeholder, phase_train_placeholder, embeddings)
    return face_encoded

def encoding_new(frame, sess, images_placeholder, phase_train_placeholder, embeddings):
    face = frame#cv2.imread(fname)
    face = misc.imresize(face, (EMBEDDING_SIZE, EMBEDDING_SIZE), interp="bilinear")
    face_prewhiten = prewhiten(face)
    face_encoded = facenet_embedding([face_prewhiten], sess, images_placeholder, phase_train_placeholder, embeddings)
    return face_encoded

def encoding_flip(fname, sess, images_placeholder, phase_train_placeholder, embeddings):
    face = cv2.imread(fname)
    face = cv2.flip(face, 1)
    face = misc.imresize(face, (EMBEDDING_SIZE, EMBEDDING_SIZE), interp="bilinear")
    face_prewhiten = prewhiten(face)
    face_encoded = facenet_embedding([face_prewhiten], sess, images_placeholder, phase_train_placeholder, embeddings)
    return face_encoded

def facenet_embedding(face_prewhiten, sess, images_placeholder, phase_train_placeholder, embeddings):
    feed_dict = {images_placeholder: face_prewhiten, phase_train_placeholder: False}
    return sess.run(embeddings, feed_dict=feed_dict)

def facenet_crop(frame, bounding_box):
    face_cropped = frame[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :]
    face_cropped = face_cropped[:, :, [2, 1, 0]]
    face = misc.imresize(face_cropped, (EMBEDDING_SIZE, EMBEDDING_SIZE), interp='bilinear')  # resize face to be used for face verification
    face_prewhiten = prewhiten(face)
    # face_prewhiten = face
    return face_prewhiten

def frame_encoding(frame, box, facenet_sess, images_placeholder, phase_train_placeholder, embeddings):
    face_prewhiten = facenet_crop(frame, box.astype(int))
    list_faces = [face_prewhiten]
    face_encoded = facenet_embedding(list_faces, facenet_sess, images_placeholder, phase_train_placeholder, embeddings)
    return face_encoded

def face_crop(frame, bounding_box):
    face_cropped = frame[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :]
    face_cropped = face_cropped[:, :, [2, 1, 0]]
    return face_cropped

def face_prewhiten(face):
    face = misc.imresize(face, (EMBEDDING_SIZE, EMBEDDING_SIZE), interp='bilinear')  # resize face to be used for face verification
    face = prewhiten(face)
    return face

def frame_encoding_flip(frame, box, facenet_sess, images_placeholder, phase_train_placeholder, embeddings):
    face_cropped = face_crop(frame, box.astype(int))
    face_flipped = cv2.flip(face_cropped, 1)
    face_cropped = face_prewhiten(face_cropped)
    face_flipped = face_prewhiten(face_flipped)
    list_cropped = [face_cropped]
    list_flipped = [face_flipped]

    encoding_cropped = facenet_embedding(list_cropped, facenet_sess, images_placeholder, phase_train_placeholder, embeddings)
    encoding_flipped = facenet_embedding(list_flipped, facenet_sess, images_placeholder, phase_train_placeholder, embeddings)
    return encoding_cropped, encoding_flipped
def load_facenet():
    """
    load Facenet model and placeholders for outputs
    :return: session, placeholder
    """

    tf.Graph().as_default()
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.Session(config=config)
    # sess = tf.Session(config=config)
    ## load facenet model
    load_model(FLAGS.model)

    ## define placeholder
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    ## output
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    return (sess, images_placeholder, phase_train_placeholder, embeddings)

class FaceEncoder():
    encoder = load_facenet()
    # def __init__(self):
    #     self.encoder = load_facenet()