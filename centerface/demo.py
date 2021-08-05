import cv2
from centerface import CenterFace


def test_image():
    frame = cv2.imread('/media/cyg/DATA1/DataSet/Face-Anti-spoofing/RITS/images/train/real/bao_real_mask_glasses_normal_5.jpg')
    h, w = frame.shape[:2]
    centerface = CenterFace()
    dets = centerface(frame)
    for det in dets:
        x1 = int(det.rect.x)
        y1 = int(det.rect.y)
        x2 = int(det.rect.x + det.rect.w)
        y2 = int(det.rect.y + det.rect.h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (2, 255, 0), 1)

        for i in range(5):
            cv2.circle(frame, (int(det.landmark[i].x), int(det.landmark[i].y)), 2, (0, 0, 255), -1)
    cv2.imshow('out', frame)
    cv2.waitKey(0)


if __name__ == '__main__':

    test_image()
