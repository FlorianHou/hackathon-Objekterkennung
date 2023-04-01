import cv2 as cv
import numpy as np
import glob
import numpy.ma as ma
import random


def getColor(bgrImg, mask, contour=None):
    hsvImg = cv.cvtColor(bgrImg, cv.COLOR_BGR2HSV)
    h = hsvImg[..., 0]
    color_picker = np.sum(h[mask == 255]) // np.sum(mask == 255)
    return color_picker


def cannyFilter(thre1, thre2):
    edge = cv.Canny(gray, thre1, thre2)
    # cv.imshow("output", np.hstack((edge, gray)))
    return edge


def nothing(*args):
    pass


def whichColor(colorH):
    """colorH: H(0-180) von HSL"""
    res = ""
    if 100 < colorH < 140:
        res = "blue"
    elif 40 < colorH < 80:
        res = "green"
    elif 160 < colorH < 180 or 0 < colorH < 20:
        res = "red"
    return res


def formErkennung(compactness):
    formung = None
    if 0.9 < compactness <= 1.2:
        formung = "kreis"
    elif 1.2 < compactness <= 1.3:
        formung = "quadrat"
    elif 1.3 < compactness <= 1.6:
        formung = "rechtEck"
    elif 1.6 < compactness <= 1.9:
        formung = "dreieck"
    elif 2.2 < compactness <= 2.5:
        formung = "stern"
    elif 4 < compactness <= 5:
        formung = "regenschirm"
    return formung


def genMask(img, points):
    __mask = np.zeros_like(img, dtype=np.uint8)
    cv.drawContours(__mask, [points], 0, 255, thickness=-1)
    __mask = cv.erode(__mask, np.ones((5, 5)), iterations=3)
    return __mask


if __name__ == "__main__":
    files = glob.glob("./imgs/*.jpg")
    for file in files:
        # Vorbereiten
        img_raw = cv.imread(file)
        img = cv.imread(file)
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        img = cv.medianBlur(img, 5)
        img = cv.bilateralFilter(img, -1, 75, 5)
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
            if M["m00"] > 2000:
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
            if formung not in ["quadrat", "rechtEck"]:
                mask = cv.bitwise_or(gray, genMask(gray, contour))
                # cv.imshow("mask", mask)

        # Color
        color_hs = []
        color_res = []
        for mask in masks:
            color_h = getColor(img_raw, mask)
            print(f"color_h(360):{color_h/255*360}")
            color_hs.append(color_h)
            color_res.append(whichColor(color_h))

        # Reihenfolge -- phase lagerStation
        cxcyList = []
        Endres = []
        EndFormung = []
        EndColorRes = []
        for contour, formung, color in zip(contours_filter4, formungRes, color_res):
            if formung in ["dreieck", "stern", "kreis", "regenschirm"]:
                M = cv.moments(contour)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cxcyList.append((cx, cy))
                Endres.append(contour)
                EndFormung.append(formung)
                EndColorRes.append(color)
        if cxcyList:
            n = len(cxcyList)
            sortIndex = np.array(cxcyList)[:, 1].argsort()
            cxcyList = np.array(cxcyList)[sortIndex]
            Endres = np.array(Endres)[sortIndex]
            EndFormung = np.array(EndFormung)[sortIndex]
            EndColorRes = np.array(EndColorRes)[sortIndex]

        ## Draw Info
        # Draw Contour
        for countour in contours_filter4:
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
        for contour, formung, color_h in zip(Endres, EndFormung, EndColorRes):
            # for contour, formung, color_h in zip(contours_filter4, formungRes, color_res):
            M = cv.moments(contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            flaeche = M["m00"]
            umfang = cv.arcLength(contour, True)
            compactness = umfang**2 / (4 * np.pi * flaeche)
            if formung not in ["quadrat", "rechtEck"]:
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
                    0.8,
                    (0, 255, 0),
                    1,
                )
                cv.putText(
                    blackGroundBGR,
                    f"compectness:{compactness:.{3}f}",
                    (cx, cy + 25),
                    cv.FONT_HERSHEY_COMPLEX,
                    0.8,
                    (0, 255, 0),
                    1,
                )
                cv.putText(
                    blackGroundBGR,
                    f"color:{color_h}",
                    (cx, cy + 50),
                    cv.FONT_HERSHEY_COMPLEX,
                    0.8,
                    (0, 255, 0),
                    1,
                )
                cv.putText(
                    blackGroundBGR,
                    f"index:{index}",
                    (cx, cy + 75),
                    cv.FONT_HERSHEY_COMPLEX,
                    0.8,
                    (0, 255, 0),
                    1,
                )
                index += 1

        cv.imshow("output", cv.resize(blackGroundBGR, (1600, 1050)))
        key = cv.waitKey(0)
        if key == ord("q"):
            cv.destroyAllWindows()
