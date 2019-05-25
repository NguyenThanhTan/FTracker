import sys
sys.path.append('/opt/hades/face_embedding/')
from face_embedding.face_encoder import FaceEncoder, encoding_new


def get_img_facenet(img):
    return encoding_new(img, *(FaceEncoder.encoder)).squeeze()
