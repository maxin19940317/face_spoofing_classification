import numpy as np
import cv2

from .object import Point, Face_Object


class CenterFace(object):
    def __init__(self):
        self.net = cv2.dnn.readNetFromONNX('./weights/centerface_weights/centerface.onnx')
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = 0, 0, 0, 0

    def __call__(self, img, threshold=0.35):
        height, width = img.shape[:2]
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = self.transform(height, width)
        return self.inference_opencv(img, threshold)

    def inference_opencv(self, img, threshold):
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(self.img_w_new, self.img_h_new), mean=(0, 0, 0),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        heatmap, scale, offset, lms = self.net.forward(["537", "538", "539", '540'])
        return self.postprocess(heatmap, lms, offset, scale, threshold)

    def transform(self, h, w):
        img_h_new, img_w_new = int(np.ceil(h / 32) * 32), int(np.ceil(w / 32) * 32)
        scale_h, scale_w = img_h_new / h, img_w_new / w
        return img_h_new, img_w_new, scale_h, scale_w

    def postprocess(self, heatmap, lms, offset, scale, threshold):
        dets = self.decode(heatmap, scale, offset, lms, (self.img_h_new, self.img_w_new), threshold=threshold)
        # if len(dets) > 0:
        #     dets[:, 0:4:2] = dets[:, 0:4:2]
        #     dets[:, 1:4:2] = dets[:, 1:4:2]
        #     dets[:, 5:15:2] = dets[:, 5:15:2] / self.scale_w
        #     dets[:, 6:15:2] = dets[:, 6:15:2] / self.scale_h
        # else:
        #     dets = np.empty(shape=[0, 15], dtype=np.float32)
        return dets

    def decode(self, heatmap, scale, offset, landmark, size, threshold=0.1):
        heatmap = np.squeeze(heatmap)
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > threshold)
        boxes = []
        if len(c0) > 0:
            for i in range(len(c0)):
                s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
                o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
                s = heatmap[c0[i], c1[i]]
                x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2), max(0, (c0[i] + o0 + 0.5) * 4 - s0 / 2)
                x1, y1 = min(x1, size[1]), min(y1, size[0])

                obj = Face_Object()
                obj.rect.x = x1 / self.scale_w
                obj.rect.y = y1 / self.scale_h
                obj.rect.w = (min(x1 + s1, size[1]) - x1) / self.scale_w
                obj.rect.h = (min(y1 + s0, size[0]) - y1) / self.scale_h
                obj.prob = s
                obj.landmark = [Point(), Point(), Point(), Point(), Point()]
                for j in range(5):
                    obj.landmark[j].x = landmark[0, j * 2 + 1, c0[i], c1[i]] * s1 + x1 / self.scale_h
                    obj.landmark[j].y = landmark[0, j * 2, c0[i], c1[i]] * s0 + y1 / self.scale_w

                boxes.append(obj)

            # boxes = np.asarray(boxes, dtype=np.float32)
            # keep = self.nms(boxes[:, :4], boxes[:, 4], 0.3)
            keep = self.nms(boxes, 0.3)
            boxes = boxes[:len(keep)]
        return boxes

    # def nms(self, boxes, scores, nms_thresh):
    #     x1 = boxes[:, 0]
    #     y1 = boxes[:, 1]
    #     x2 = boxes[:, 2]
    #     y2 = boxes[:, 3]
    #     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #     order = np.argsort(scores)[::-1]
    #     num_detections = boxes.shape[0]
    #     suppressed = np.zeros((num_detections,), dtype=np.bool)
    #
    #     keep = []
    #     for _i in range(num_detections):
    #         i = order[_i]
    #         if suppressed[i]:
    #             continue
    #         keep.append(i)
    #
    #         ix1 = x1[i]
    #         iy1 = y1[i]
    #         ix2 = x2[i]
    #         iy2 = y2[i]
    #         iarea = areas[i]
    #
    #         for _j in range(_i + 1, num_detections):
    #             j = order[_j]
    #             if suppressed[j]:
    #                 continue
    #
    #             xx1 = max(ix1, x1[j])
    #             yy1 = max(iy1, y1[j])
    #             xx2 = min(ix2, x2[j])
    #             yy2 = min(iy2, y2[j])
    #             w = max(0, xx2 - xx1 + 1)
    #             h = max(0, yy2 - yy1 + 1)
    #
    #             inter = w * h
    #             ovr = inter / (iarea + areas[j] - inter)
    #             if ovr >= nms_thresh:
    #                 suppressed[j] = True
    #
    #     return keep

    def nms(self, faceobjects, nms_threshold):
        picked = []

        n = len(faceobjects)

        areas = []
        for i in range(n):
            areas.append(faceobjects[i].rect.area())

        for i in range(n):
            a = faceobjects[i]

            keep = True
            for j in range(len(picked)):
                b = faceobjects[picked[j]]

                # intersection over union
                inter_area = a.rect.intersection_area(b.rect)
                union_area = areas[i] + areas[picked[j]] - inter_area
                # float IoU = inter_area / union_area
                if inter_area / union_area > nms_threshold:
                    keep = False

            if keep:
                picked.append(i)

        return picked
