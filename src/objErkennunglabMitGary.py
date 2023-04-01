import rospy
import cv2 as cv
import random
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from enum import Enum


def objErkennung(imgData):
    """Code fuer ObjErkennung"""

    def getColor(bgrImg, __mask, contour=None):
        hsvImg = cv.cvtColor(bgrImg, cv.COLOR_BGR2HSV)
        h = hsvImg[..., 0]
        color_picker = np.sum(h[__mask == 255]) // np.sum(__mask == 255)
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
        elif 22.5 < colorH < 32.5:
            res = "gelb"
        else:
            res = "nopass"
        return res

    def formErkennung(compactness, ratio, fillfactor):
        formung = ""
        if 1 < compactness <= 1.15:
            formung = "kreis"
        elif 1.2 < compactness <= 1.34 and 0.9 < ratio < 1.5:
            formung = "quadrat"
        elif 1.37 < compactness <= 1.55:
            formung = "rechtEck"
        elif 1.56 < compactness <= 1.68 and ratio < 1.2:
            formung = "dreieck"
        elif 1.8 < compactness <= 2.6:
            formung = "stern"
        elif 3.2 < compactness <= 3.6:
            formung = "regenschirm"
        else:
            formung = "nopass"
        return formung

    def genMask(img, points):
        __mask = np.zeros_like(img, dtype=np.uint8)
        cv.drawContours(__mask, [points], 0, 255, thickness=-1)
        __mask = cv.erode(__mask, np.ones((5, 5)), iterations=1)
        return __mask

    def reihenfolge(contours, forms, colors):
        cxcyList = []

        for contour, form, color in zip(contours, forms, colors):
            M = cv.moments(contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cxcyList.append((cx, cy))
        if cxcyList:
            sortIndex = np.array(cxcyList)[:, 1].argsort()
            contours = np.array(contours)[sortIndex]
            forms = np.array(forms)[sortIndex]
            colors = np.array(colors)[sortIndex]
        return contours, forms, colors

    def genRes(contours, forms, colors):
        res = []
        for index, (contour, form, color) in enumerate(zip(contours, forms, colors)):
            res.append({"formung": form, "index": index, "color": color})
        return res

    def draw(contours, forms, colors, windowName):
        blackGround = np.zeros_like(gray, dtype=np.uint8)
        blackGroundBGR = cv.cvtColor(blackGround.copy(), cv.COLOR_GRAY2BGR)
        for index, (contour, form, color) in enumerate(zip(contours, forms, colors)):
            # for contour, formung, color_h in zip(contours_filter4, formungRes, color_res):
            colorContour = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            blackGroundBGR = cv.drawContours(
                blackGroundBGR,
                [contour],
                0,
                colorContour,
                3,
            )
            M = cv.moments(contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            flaeche = M["m00"]
            umfang = cv.arcLength(contour, True)
            compactness = umfang**2 / (4 * np.pi * flaeche)
            cv.circle(
                blackGroundBGR,
                (cx, cy),
                radius=int(umfang * 0.01),
                color=(0, 0, 255),
                thickness=-1,
            )
            cv.putText(
                blackGroundBGR,
                f"formung:{form}",
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
                f"color:{color}",
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
        cv.imshow(windowName, blackGroundBGR)
        cv.imshow("gray", gray)

    suchFormungStation = ["quadrat"]
    suchFormungMenu = ["kreis", "stern", "dreieck", "regenschirm"]

    # Vorbereiten
    img_orig = imgData
    img = img_orig.copy()
    img = cv.medianBlur(img, 5)
    img = cv.bilateralFilter(img, -1, 25, 15)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # img = cv.GaussianBlur(img_orig.copy(), (9, 9), 0)
    # img = cv.bilateralFilter(img, -1, 75, 15)
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    a = lab[..., 1]
    b = lab[..., 2]
    ret, redChannel = cv.threshold(a, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # ret, greenChannel = cv.threshold(a, 128, 255, cv.THRESH_BINARY_INV)
    # ret, blueChannel = cv.threshold(b, 128, 255, cv.THRESH_BINARY_INV)
    ret, gelbChannel = cv.threshold(b, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    gray = cv.equalizeHist(gray)
    edge1 = cv.Canny(redChannel, 100, 300)
    edge2 = cv.Canny(gelbChannel, 100, 300)
    edge3 = cv.Canny(gray, 100, 300)
    edge = edge1 + edge2 + edge3
    # edge = cv.Canny(gray, 300, 600)
    cv.imshow("edge", edge)

    # alaplacian = cv.Laplacian(a, cv.CV_64F)
    # blaplacian = cv.Laplacian(b, cv.CV_64F)
    # cv.imshow("greenChannel", greenChannel)
    # cv.imshow("blueChannel", blueChannel)
    cv.imshow("redChannel", redChannel)
    cv.imshow("gelbChannel", gelbChannel)

    edge = cv.morphologyEx(
        edge,
        cv.MORPH_CLOSE,
        kernel=cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)),
        # iterations=3,
    )

    # blackGround = np.zeros_like(gray, dtype=np.uint8)
    # blackGroundBGR = cv.cvtColor(blackGround.copy(), cv.COLOR_GRAY2BGR)

    contours, hierach = cv.findContours(edge, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contourIMGVorFilter = np.zeros_like(gray, np.uint8)
    contourIMGVorFilter = cv.drawContours(
        contourIMGVorFilter, contours, -1, 255, thickness=3
    )
    cv.imshow("origContours", contourIMGVorFilter)
    # Flaeche
    contoursRes = []
    old_center = (0, 0)
    formungRes = []
    for contour in contours:
        M = cv.moments(contour)
        if M["m00"] > 600:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            if (old_center[0] - cx) ** 2 + (old_center[1] - cy) ** 2 > 4:
                flaeche = M["m00"]
                umfang = cv.arcLength(contour, True)
                minArea = cv.minAreaRect(contour)
                ratio = max(minArea[1]) / min(minArea[1])

                fillFactor = flaeche / (minArea[1][0] * minArea[1][1])
                if fillFactor > 0.3 and ratio < 2:
                    contoursRes.append(contour)
                    # if len(contoursRes)>1:
                    #     for contourInRes in contoursRes[:-1]:
                    #         M2 = cv.moments(contourInRes)
                    #         cx2 = int(M2["m10"] / M2["m00"])
                    #         cy2 = int(M2["m01"] / M2["m00"])
                    #         if (cx2 - cx) ** 2 + (cy2 - cy) ** 2 < 10:
                    #             print("pop!!")
                    #             formungRes.pop()
                    #             contoursRes.pop()
                    #             break
    HasNear = [False for _ in range(len(contoursRes))]
    for m, contour1 in enumerate(contoursRes):
        M1 = cv.moments(contour1)
        cx1 = int(M1["m10"] / M1["m00"])
        cy1 = int(M1["m01"] / M1["m00"])
        if not HasNear[m]:
            for n, contour2 in enumerate(contoursRes[(m + 1) :]):
                M2 = cv.moments(contour2)
                cx2 = int(M2["m10"] / M2["m00"])
                cy2 = int(M2["m01"] / M2["m00"])
                if (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2 < 10:
                    HasNear[m + n + 1] = True
    temRes = []

    contourIMGVorFilter = np.zeros_like(gray, np.uint8)
    contourIMGVorFilter = cv.drawContours(
        contourIMGVorFilter, contoursRes, -1, 255, thickness=3
    )
    cv.imshow("contourIMGVorFilter", contourIMGVorFilter)
    for i, contour in enumerate(contoursRes):
        if not HasNear[i]:
            M = cv.moments(contour)
            flaeche = M["m00"]
            umfang = cv.arcLength(contour, True)
            compactness = umfang**2 / (4 * np.pi * flaeche)
            minArea = cv.minAreaRect(contour)
            ratio = max(minArea[1]) / min(minArea[1])
            fillFactor = flaeche / (minArea[1][0] * minArea[1][1])
            formungRes.append(formErkennung(compactness, ratio, fillFactor))
            temRes.append(contour)
    contoursRes = temRes.copy()
    contourIMGNachFilter = np.zeros_like(gray, np.uint8)
    contourIMGNachFilter = cv.drawContours(
        contourIMGVorFilter, contoursRes, -1, 255, thickness=3
    )
    cv.imshow("contourIMGNachFilter", contourIMGNachFilter)
    # Mask
    contoursStation = []
    contoursMenu = []
    formsStation = []
    formsMenu = []
    masksStation = []
    masksMenu = []
    colorsStation = []
    colorsMenu = []
    for contour, formung in zip(contoursRes, formungRes):
        if formung in suchFormungMenu:
            masksMenu.append(genMask(gray, contour))
            # color = getColor(img_orig, masksMenu[-1])
            colorsMenu.append(whichColor(getColor(img_orig, masksMenu[-1])))
            contoursMenu.append(contour)
            formsMenu.append(formung)
        if formung in suchFormungStation:
            masksStation.append(genMask(gray, contour))
            colorsStation.append(whichColor(getColor(img_orig, masksStation[-1])))
            contoursStation.append(contour)
            formsStation.append(formung)
    mask_all = np.zeros_like(gray, dtype=np.uint8)
    for mask in masksMenu:
        mask_all = np.bitwise_or(mask_all, mask)
    for mask in masksStation:
        mask_all = np.bitwise_or(mask_all, mask)
    # cv.imshow('mask_all',mask_all)

    # Reihenfolge
    contoursStation, formsStation, colorsStation = reihenfolge(
        contoursStation, formsStation, colorsStation
    )
    contoursMenu, formsMenu, colorsMenu = reihenfolge(
        contoursMenu, formsMenu, colorsMenu
    )
    # Draw
    draw(contoursStation, formsStation, colorsStation, "Station")
    draw(contoursMenu, formsMenu, colorsMenu, "Menu")
    cv.waitKey(10)

    # Res
    ResMenu = genRes(contoursMenu, formsMenu, colorsMenu)
    ResStation = genRes(contoursStation, formsStation, colorsStation)
    return ResMenu, ResStation


class Formung(Enum):
    kreis = 0
    dreieck = 1
    stern = 2
    regenschirm = 3
    quadrat = 4
    nopass = -1


class Color(Enum):
    red = 0
    green = 1
    blue = 2
    gelb = 3
    nopass = -1


def genMsg(resList):
    msg = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    for res in resList:
        msg[res["index"]] = Formung[res["formung"].lower()].value
        msg[res["index"] + 5] = Color[res["color"].lower()].value
    return msg


counter = 0
msgMenuList = []
msgStationList = []


def callback(imgData, pub):

    (pubMenu, pubStation) = pub
    print(imgData.header)
    print(imgData.encoding)
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(imgData, "bgr8")
    ResMenu, ResStation = objErkennung(img)

    msgMenu = genMsg(ResMenu)
    msgStation = genMsg(ResStation)
    if len(msgMenuList) == 3:
        if msgMenuList[0] == msgMenuList[1] == msgMenuList[2]:
            # if msgMenuList[0][0]!=-1:
            #     pubMenu.publish(Int32MultiArray(data=msgMenuList[0]))
            print(f"msgMenu:{msgMenu}\n")
            msgMenuList.reverse()
            msgMenuList.pop()
            msgMenuList.reverse()
        else:
            msgMenuList.reverse()
            msgMenuList.pop()
            msgMenuList.reverse()
    else:
        msgMenuList.append(msgMenu)

    if len(msgStationList) == 3:
        if msgStationList[0] == msgStationList[1] == msgStationList[2]:
            # if msgStationList[0][0]!=-1:
            #     pubStation.publish(Int32MultiArray(data=msgStationList[0]))
            print(f"msgStation:{msgStation}\n")
            msgStationList.reverse()
            msgStationList.pop()
            msgStationList.reverse()
        else:
            msgStationList.reverse()
            msgStationList.pop()
            msgStationList.reverse()
    else:
        msgStationList.append(msgStation)


def main():
    print("start")
    rospy.init_node("ObjErkennung", anonymous=True)
    pubMenu = rospy.Publisher("ResMenu", Int32MultiArray, queue_size=1000)
    pubStation = rospy.Publisher("ResStation", Int32MultiArray, queue_size=1000)
    print("sub")
    rospy.Subscriber(
        "/cam_front/color/image_raw",
        Image,
        callback=callback,
        callback_args=(pubMenu, pubStation),
    )
    print("spain")
    rospy.spin()


if __name__ == "__main__":
    main()
