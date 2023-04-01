import cv2 as cv
import numpy as np
import glob
import numpy.ma as ma


def cvtJPG():
    pass


def cannyFilter(thre1, thre2):
    edge = cv.Canny(gray, thre1, thre2)
    # cv.imshow("output", np.hstack((edge, gray)))
    return edge


def nothing(*args):
    pass


def genMask(img, points):
    mask = np.zeros_like(img, dtype=np.uint8)
    cv.fillConvexPoly(mask, points, 1)
    # mask = cv.erode(mask, np.ones((20, 20)))
    return mask


if __name__ == "__main__":
    files = glob.glob("./imgs/*.jpg")
    for file in files:
        # Vorbereiten
        img = cv.imread(file)
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # img = cv.medianBlur(img, 5)
        # img = cv.bilateralFilter(img, -1, 75, 5)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        # kontours = cv.Canny(gray, 3, 3)
        cv.namedWindow("output")
        cv.createTrackbar("Schwellenwert1", "output", 0, 600, nothing)
        cv.setTrackbarPos("Schwellenwert1", "output", 200)
        cv.createTrackbar("Schwellenwert2", "output", 0, 600, nothing)
        cv.setTrackbarPos("Schwellenwert2", "output", 400)
        cv.createTrackbar("fillFactor", "output", 0, 100, nothing)
        cv.setTrackbarPos("fillFactor", "output", 50)
        cv.createTrackbar("minArea", "output", 0, 2000, nothing)
        cv.setTrackbarPos("minArea", "output", 600)
        cv.createTrackbar("compactness", "output", 0, 300, nothing)
        cv.setTrackbarPos("compactness", "output", 50)
        cv.createTrackbar("d", "output", 0, 100000, nothing)
        cv.setTrackbarPos("d", "output", 50)
        while True:
            thre1Val = cv.getTrackbarPos("Schwellenwert1", "output")
            thre2Val = cv.getTrackbarPos("Schwellenwert2", "output")
            minfillFactorVal = cv.getTrackbarPos("fillFactor", "output")
            minimaleflaecheVal = cv.getTrackbarPos("minArea", "output")
            compactnessVal = cv.getTrackbarPos("compactness", "output")
            dVal = cv.getTrackbarPos("d", "output")
            # Canny
            edge = cannyFilter(thre1Val, thre2Val)
            # Kontour
            kontours, hiera = cv.findContours(
                edge.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE 
            )

            # Zeichen Kontour
            grayCopy = gray.copy()
            imgCopy = img.copy()
            gray2BRG = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
            # cv.drawContours(gray2BRG, kontours, -1, (0, 0, 255), 3)

            # fill Faktor&Compactness
            kontour_filter = []
            for kontour in kontours:
                M = cv.moments(kontour)

                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    flaeche = M["m00"]
                    umfang = cv.arcLength(kontour, True)
                    minArea = cv.minAreaRect(kontour)
                    # rectArea = cv.re
                    fillFactor = flaeche / (minArea[1][0] * minArea[1][1])
                    compactness = umfang**2 / (4 * np.pi * flaeche)
                    if (
                        M["m00"] >= minimaleflaecheVal
                        and compactness >= compactnessVal / 100
                    ):
                        kontour_filter.append(kontour)
                        # cv.putText(
                        #     gray2BRG,
                        #     f"compactness:{compactness:.{3}f}",
                        #     (cx, cy),
                        #     cv.FONT_HERSHEY_COMPLEX,
                        #     0.8,
                        #     (255, 0, 0),
                        #     1,
                        # )
            kontour_filter2 = []
            tmp_center = {"cx": 0, "cy": 0}
            for kontour in kontour_filter:
                M = cv.moments(kontour)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                d = (cx - tmp_center["cx"]) ** 2 + (cy - tmp_center["cy"]) ** 2
                if d > dVal:
                    # TODO
                    flaeche = M["m00"]
                    umfang = cv.arcLength(kontour, True)
                    minArea = cv.minAreaRect(kontour)
                    fillFactor = flaeche / (minArea[1][0] * minArea[1][1])
                    compactness = umfang**2 / (4 * np.pi * flaeche)
                    formung = None
                    if 0.9 < compactness < 1.2:
                        formung = "kreis"
                    elif 1.2 < compactness < 1.3:
                        formung = "quadrat"
                    elif 1.6 < compactness < 1.9:
                        formung = "dreieck"
                    elif 2.2 < compactness < 2.5:
                        formung = "stern"
                    elif 4 < compactness < 5:
                        formung = "regenschirm"

                    cv.putText(
                        gray2BRG,
                        f"compactness:{compactness:.{3}f}, formung: {formung}",
                        (cx, cy),
                        cv.FONT_HERSHEY_COMPLEX,
                        0.8,
                        (0, 255, 255),
                        1,
                    )

                    kontour_filter2.append(
                        {
                            "kontour": kontour,
                            "compactness": compactness,
                            "center": (cx, cy),
                            "formung": formung,
                        }
                    )

                tmp_center["cx"] = cx
                tmp_center["cy"] = cy
            # masks:
            mask_or = np.zeros_like(gray)
            for kontour in kontour_filter2:
                mask = genMask(gray, kontour["kontour"]).astype(np.bool_)
                h = hsv[..., 0]

                masked_array = ma.masked_array(h, mask=mask)
                color_avr = np.sum(h.astype(np.uint8) * mask.astype(np.uint8)) / np.sum(mask)
                ones_masked_array = np.sum((h.astype(np.uint8) * mask.astype(np.uint8)) != 0)
                ones_mask = np.sum((mask) != 0)
                mask_or = np.bitwise_or(mask_or, mask)
                # cv.imshow("outpu2", mask.astype(np.uint8)*255)
                # key = cv.waitKey(1000)
                # if key == ord("q"):
                #     cv.destroyAllWindows()
                print(f"{kontour['formung']}, {color_avr}")

            # Zeigen
            cv.drawContours(
                gray2BRG,
                [val["kontour"] for val in kontour_filter2],
                -1,
                (0, 0, 255),
                6,
            )
            cv.namedWindow("output", cv.WINDOW_AUTOSIZE)
            h, w, *_ = img.shape
            # cv.imshow(
            #     "output",
            #     cv.resize(np.hstack((hsv[..., 2], gray)), (int(w / 2), int(h / 2))),
            # )

            cv.imshow(
                "output",
                cv.resize(
                    np.hstack((cv.cvtColor(edge, cv.COLOR_GRAY2BGR), gray2BRG)),
                    (int(w), int(h / 2)),
                    interpolation=cv.INTER_CUBIC,
                ),
            )
            # cv.imshow("output", cv.resize(mask_or*255,(w//2,h//2)))

            # cv.imshow("output", cv.resize(gray, (int(w / 2), int(h / 2))))
            key = cv.waitKey(10)
            if key == ord("q"):
                cv.destroyAllWindows()
                break
