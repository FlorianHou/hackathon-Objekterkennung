import rospy
import cv2 as cv
import random
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Int8MultiArray
from enum import Enum


def objErkennung(imgData):
    """Code fuer ObjErkennung"""

    def getColor(bgrImg, mask, contour=None):
        hsvImg = cv.cvtColor(bgrImg, cv.COLOR_BGR2HSV)
        h = hsvImg[..., 0]
        color_picker = np.sum(h[mask == 255]) // np.sum(mask == 255)
        return color_picker

    def whichColor(colorH):
        """colorH: H(0-180) von HSL"""
        res = ""
        if 80 < colorH < 150:
            res = "blue"
        elif 40 < colorH < 80:
            res = "green"
        elif 160 < colorH < 180 or 0 < colorH < 20:
            res = "red"
        return res

    def formErkennung(compactness):
        formung = None
        if 0.9 < compactness <= 1.15:
            formung = "kreis"
        elif 1.28 < compactness <= 1.31:
            formung = "quadrat"
        elif 1.37 < compactness <= 1.6:
            formung = "rechtEck"
        elif 1.6 < compactness <= 1.7:
            formung = "dreieck"
        elif 1.8 < compactness <= 2.1:
            formung = "stern"
        elif 4 < compactness <= 5:
            formung = "regenschirm"
        return formung

    def genMask(img, points):
        __mask = np.zeros_like(img, dtype=np.uint8)
        cv.drawContours(__mask, [points], 0, 255, thickness=-1)
        __mask = cv.erode(__mask, np.ones((5, 5)), iterations=3)
        return __mask

    # Vorbereiten
    img_orig = imgData
    img = img_orig.copy()
    # hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # img = cv.medianBlur(img, 5)
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    a = lab[..., 1]
    b = lab[..., 2]
    img = cv.bilateralFilter(img, -1, 75, 5)
    cv.imshow("lab:a", a)
    cv.imshow("lab:b", b)
    img = cv.bilateralFilter(img, -1, 75, 5)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    edge = cv.Canny(a, 100, 300)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    edge = cv.Canny(gray, 100, 300)

    edge = cv.morphologyEx(
        edge,
        cv.MORPH_CLOSE,
        kernel=cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)),
        iterations=3,
    )
    blackGround = np.zeros_like(gray, dtype=np.uint8)
    blackGroundBGR = cv.cvtColor(blackGround.copy() * 255, cv.COLOR_GRAY2BGR)

    contours, hierach = cv.findContours(edge, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours_filter1 = []
    # nicht linie
    for countour in contours:
        M = cv.moments(countour)
        flaeche = M["m00"]
        umfang = cv.arcLength(countour, True)
        if M["m00"] > 0:
            contours_filter1.append(countour)

    # Flaeche
    contours_filter2 = []
    for countour in contours_filter1:
        M = cv.moments(countour)
        flaeche = M["m00"]
        umfang = cv.arcLength(countour, True)
        if M["m00"] > 900:
            contours_filter2.append(countour)

    # Center
    contours_filter3 = []
    old_center = (0, 0)
    old_falache = 0
    for contour in contours_filter2:
        M = cv.moments(contour)
        flaeche = M["m00"]
        umfang = cv.arcLength(contour, True)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if (old_center[0] - cx) ** 2 + (old_center[1] - cy) ** 2 > 2:
            contours_filter3.append(contour)
        old_center = (cx, cy)
        old_falache = flaeche

    # Formung Erkennung
    contours_filter4 = []
    formungRes = []
    for contour in contours_filter3:
        M = cv.moments(contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        flaeche = M["m00"]
        umfang = cv.arcLength(contour, True)
        minArea = cv.minAreaRect(contour)
        fillFactor = flaeche / (minArea[1][0] * minArea[1][1])
        ratio = max(minArea[1]) / min(minArea[1])
        compactness = umfang**2 / (4 * np.pi * flaeche)
        if fillFactor > 0.4 and ratio < 2:
            contours_filter4.append(contour)
            formung = formErkennung(compactness)
            formungRes.append(formung)

    # Mask
    mask = np.zeros_like(gray, dtype=np.uint8)
    masks = []
    for contour, formung in zip(contours_filter4, formungRes):
        mask = genMask(gray, contour)
        masks.append(genMask(mask, contour))
        print(np.sum(mask))
        # if formung not in ["quadrat", "rechtEck"]:
        if formung not in ["rechtEck", None]:
            mask = cv.bitwise_or(gray, genMask(gray, contour))
            # cv.imshow("mask", mask)

    # Color
    color_hs = []
    color_res = []
    for mask in masks:
        color_h = getColor(img_orig, mask)
        print(f"color_h(360):{color_h/180*360}")
        color_hs.append(color_h)
        color_res.append(whichColor(color_h))

    # Reihenfolge -- phase lagerStation
    cxcyList = []
    Endres = []
    EndFormung = []
    EndColorRes = []
    for contour, formung, color in zip(contours_filter4, formungRes, color_res):
        if formung in ["dreieck", "stern", "kreis", "regenschirm", "quadrat"]:
            M = cv.moments(contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cxcyList.append((cx, cy))
            Endres.append(contour)
            EndFormung.append(formung)
            EndColorRes.append(color)
            if formung == "stern":
                print(f"Sterncolor:{color}")
    if cxcyList:
        sortIndex = np.array(cxcyList)[:, 1].argsort()
        cxcyList = np.array(cxcyList)[sortIndex]
        Endres = np.array(Endres)[sortIndex]
        EndFormung = np.array(EndFormung)[sortIndex]
        EndColorRes = np.array(EndColorRes)[sortIndex]

    ## Draw Info
    # Draw Contour
    for countour in contours_filter1:
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        blackGroundBGR = cv.drawContours(
            blackGroundBGR,
            [countour],
            0,
            color,
            3,
        )

    # Draw Text &Center &color
    index = 0
    res = []
    for contour, formung, color_h in zip(Endres, EndFormung, EndColorRes):
        # for contour, formung, color_h in zip(contours_filter4, formungRes, color_res):
        M = cv.moments(contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        flaeche = M["m00"]
        umfang = cv.arcLength(contour, True)
        compactness = umfang**2 / (4 * np.pi * flaeche)
        if formung not in ["rechtEck", None]:
            cv.circle(
                blackGroundBGR,
                (cx, cy),
                radius=int(umfang * 0.01),
                color=(0, 0, 255),
                thickness=-1,
            )
            cv.putText(
                blackGroundBGR,
                f"formung:{formung}",
                (cx, cy),
                cv.FONT_HERSHEY_COMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )
            cv.putText(
                blackGroundBGR,
                f"compectness:{compactness:.{3}f}",
                (cx, cy + 15),
                cv.FONT_HERSHEY_COMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )
            cv.putText(
                blackGroundBGR,
                f"color:{color_h}",
                (cx, cy + 30),
                cv.FONT_HERSHEY_COMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )
            cv.putText(
                blackGroundBGR,
                f"index:{index}",
                (cx, cy + 45),
                cv.FONT_HERSHEY_COMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )
            index += 1
        res.append({"formung": formung, "index": index - 1, "color": color_h})

    cv.imshow("output", blackGroundBGR)
    cv.imshow("output2", gray)
    key = cv.waitKey(5)
    # if key == ord("q"):
    #     cv.destroyAllWindows()
    return res


class Formung(Enum):
    kreis = 0
    dreieck = 1
    stern = 2
    regenschirm = 3
    quadrat = 4


class Color(Enum):
    red = 0
    green = 1
    blue = 2


def genMsg(resList):
    msg = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    for res in resList:
        msg[res["index"]] = Formung[res["formung"].lower()].value
        msg[res["index"] + 5] = Color[res["color"].lower()].value
    return msg


def callback(imgData, pub):
    print(imgData.header)
    print(imgData.encoding)
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(imgData, "bgr8")
    # cv.imshow("output", img)
    # cv.waitKey(10)
    res = objErkennung(img)
    msg = genMsg(res)
    pub.publish(Int8MultiArray(data=msg))
    print(res, msg)


def main():
    rospy.init_node("ObjktErkennung", anonymous=True)
    pub = rospy.Publisher("ResObjErkennung", Int8MultiArray)
    rospy.Subscriber(
        "/cam_front/color/image_raw", Image, callback=callback, callback_args=pub
    )
    rospy.spin()


if __name__ == "__main__":
    res = []
    main()
