import settings
import torch
import torch.backends.cudnn as cudnn
import time
from .utils.layers.functions.prior_box import PriorBox
from .utils.nms_wrapper import nms
from .utils.models.faceboxes import FaceBoxes
from .utils.box_utils import decode
from .utils.timer import Timer
from .utils.config import cfg
import cv2
import numpy as np


class Detector:
    """
        Detector for predicting face bouding boxes
    """
    def __init__(self, model_path, logger=None):
        # load models
        self.gpu = settings.USE_GPU
        self.min_score = settings.CONFIDENCE_THRES
        print(settings.CONFIDENCE_THRES)
        self.input_size = settings.INPUT_SIZE
        self.scale = settings.SCALE_UP
        self.device = torch.device("cuda" if self.gpu else "cpu")
        self.net = self.load_net(model_path)

    def load_net(self, model_path):
        # set device and mode
        def check_keys(model, pretrained_state_dict):
            ckpt_keys = set(pretrained_state_dict.keys())
            model_keys = set(model.state_dict().keys())
            used_pretrained_keys = model_keys & ckpt_keys
            unused_pretrained_keys = ckpt_keys - model_keys
            missing_keys = model_keys - ckpt_keys
            print('Missing keys:{}'.format(len(missing_keys)))
            print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
            print('Used keys:{}'.format(len(used_pretrained_keys)))
            assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
            return True

        def remove_prefix(state_dict, prefix):
            ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
            print('remove prefix \'{}\''.format(prefix))
            f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
            return {f(key): value for key, value in state_dict.items()}

        def load_model(model, pretrained_path, load_to_cpu):
            print('Loading pretrained model from {}'.format(pretrained_path))
            device = torch.cuda.current_device()
            if load_to_cpu:
                pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
                pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
            else:
                device = torch.cuda.current_device()
                pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

            if "state_dict" in pretrained_dict.keys():
                pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
            else:
                pretrained_dict = remove_prefix(pretrained_dict, 'module.')
            check_keys(model, pretrained_dict)
            model.load_state_dict(pretrained_dict, strict=False)
            return model

        net = FaceBoxes(phase='test', size=None, num_classes=2)  # initialize detector
        net = load_model(net, model_path, settings.USE_GPU)
        net.eval()
        print('Finished loading model!')
        print(net)
        cudnn.benchmark = True
        net = net.to(self.device)

        return net

    def scale_image(self, xmin, ymin, xmax, ymax):
        w = self.scale*(xmax-xmin)
        h = self.scale*(ymax-ymin)
        xmin -= w
        xmax += w
        ymin -= h
        ymax += h
        return xmin, ymin, xmax, ymax

    def predict_image(self, img):
        start_time = time.perf_counter()
        boxes = self.infer(img)
        stop_time = time.perf_counter()
        result = {"duration": stop_time - start_time, "geometries":[]}

        for box in boxes:
            result["geometries"].append(box)
        return result

    def infer(self, img, path=False):
        height = img.shape[0]
        width = img.shape[1]
        im_shrink = self.input_size/max(height, width)
        if im_shrink != 1:
            img = cv2.resize(img, None, None, fx=im_shrink, fy=im_shrink, interpolation=cv2.INTER_LINEAR)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img = np.float32(img)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        out = self.net(img)  # forward pass
        priorbox = PriorBox(cfg, out[2], (im_height, im_width), phase='test')
        priors = priorbox.forward()
        priors = priors.to(self.device)
        loc, conf, _ = out
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / im_shrink
        boxes = boxes.cpu().numpy()
        scores = conf.data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > self.min_score)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:5000]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(dets, 0.3, force_cpu=True)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:750, :]
        # dets[:, 2] -= dets[:, 0]
        # dets[:, 3] -= dets[:, 1]
        dets = np.clip(dets, a_min=0, a_max=None)
        det_boxes = []
        for det in dets:
            det_boxes.append([det[0], det[1], det[2], det[3], det[4]])
        return np.array(det_boxes)

    def predict_batch(self, images):
        start_time = time.perf_counter()
        list_boxes = self.predict_images(images)
        stop_time = time.perf_counter()

        results = {"duration": (stop_time - start_time), "list_geometries": []}
        for boxes in list_boxes:
            result = {"duration": (stop_time - start_time) / len(list_boxes), "geometries": []}
            for box in boxes:
                result["geometries"].append(box)
            results['list_geometries'].append(result)
        return results

    def predict_images(self, images):
        height = images[0].shape[0]
        width = images[0].shape[1]
        im_shrink = self.input_size/max(height, width)
        if im_shrink != 1:
            for idx, img in enumerate(images):
                images[idx] = cv2.resize(img, None, None, fx=im_shrink, fy=im_shrink, interpolation=cv2.INTER_LINEAR)

        im_height, im_width, _ = images[0].shape
        scale = torch.Tensor([images[0].shape[1], images[0].shape[0], images[0].shape[1], images[0].shape[0]])
        images = np.float32(images)
        images -= np.reshape(a=[104, 117, 123], newshape=(1, 1, 1, 3))
        images = images.transpose(0, 3, 1, 2)
        images = torch.from_numpy(images)  #.unsqueeze(0)
        images = images.to(self.device)
        scale = scale.to(self.device)

        out = self.net(images)  # forward pass
        priorbox = PriorBox(cfg, out[2], (im_height, im_width), phase='test')
        priors = priorbox.forward()
        priors = priors.to(self.device)
        locs, confs, _ = out
        confs = confs.reshape((locs.shape[0], -1, 2))
        prior_data = priors.data
        batch_results = []
        for loc, conf in zip(locs, confs):
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / im_shrink
            boxes = boxes.cpu().numpy()
            scores = conf.data.cpu().numpy()[:, 1]

            # ignore low scores
            inds = np.where(scores > 0.5)[0]
            boxes = boxes[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:5000]
            boxes = boxes[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = nms(dets, 0.3, force_cpu=True)
            dets = dets[keep, :]

            # keep top-K faster NMS
            dets = dets[:750, :]
            dets[:, 2] -= dets[:, 0]
            dets[:, 3] -= dets[:, 1]
            dets = np.clip(dets, a_min=0, a_max=None)
            det_boxes = []
            for det in dets:
                det_boxes.append(((det[0], det[1], det[2], det[3]), det[4]))
            batch_results.append(det_boxes)
        return batch_results
