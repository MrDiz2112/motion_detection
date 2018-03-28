import numpy as np
import cv2


class ImageOperations:
    KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    @classmethod
    def Filter(self, frame):
        return cv2.blur(frame, (5, 5))

    @classmethod
    def DiffImg(self, t0, t1):
        img = cv2.absdiff(t0, t1)
        return img

    @classmethod
    def Threshold(self, img):
        return cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)[1]

    @classmethod
    def Morph(self, img):
        openning = cv2.morphologyEx(img, cv2.MORPH_OPEN, self.KERNEL)
        closing = cv2.morphologyEx(openning, cv2.MORPH_CLOSE, self.KERNEL)
        dilate = cv2.dilate(closing, self.KERNEL, iterations=6)
        erode = cv2.erode(dilate, self.KERNEL, iterations=4)
        openning = cv2.morphologyEx(erode, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        return openning

    @classmethod
    def CreateBackground(self, dif1, dif2, frame):
        movImg = cv2.bitwise_and(dif1, dif2)
        backImg = self.DiffImg(frame, movImg)
        return backImg

    @classmethod
    def ConvertToGray(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    @classmethod
    def CreateMovingObject(self, f0, f1, f2):
        f0 = self.Filter(f0)
        f1 = self.Filter(f1)
        f2 = self.Filter(f2)

        # Разница между кадрами
        dif1 = self.Threshold(self.DiffImg(f0, f1))
        dif2 = self.Threshold(self.DiffImg(f1, f2))

        # Движущийся объект и задник
        movImg = cv2.bitwise_and(dif1, dif2)
        backImg = self.DiffImg(f1, movImg)

        # Пересмотренный задник
        refImg = cv2.bitwise_or(f1, backImg)

        # Движущийся объект
        movObject = self.DiffImg(f1, refImg)
        movObject = self.Threshold(movObject)
        movObject = self.Morph(movObject)

        return movObject

    @classmethod
    def ConnectNearbyContours(self, contours_big, contourDist):
        decision_list = np.zeros((len(contours_big)), np.bool)
        for i in range(len(contours_big) - 1):
            cnt1 = contours_big[i]
            cnt2 = contours_big[i + 1]

            M1 = cv2.moments(cnt1)
            cx1 = int(M1['m10'] / M1['m00'])
            cy1 = int(M1['m01'] / M1['m00'])

            M2 = cv2.moments(cnt2)
            cx2 = int(M2['m10'] / M2['m00'])
            cy2 = int(M2['m01'] / M2['m00'])

            r1 = cv2.minEnclosingCircle(cnt1)[1]
            r2 = cv2.minEnclosingCircle(cnt2)[1]

            dist = np.linalg.norm(np.array((cx1, cy1)) - np.array((cx2, cy2)))

            if (dist < contourDist):
                decision_list[i + 1] = True

            contours_complete = []
            if len(contours_big) != 0:
                if (len(contours_big) == 1):
                    contours_complete = contours_big
                else:
                    connection = contours_big[0]
                    for i in range(1, len(decision_list)):
                        if decision_list[i]:
                            connection = np.vstack((connection, contours_big[i]))
                        else:
                            contours_complete.append(connection)
                            connection = contours_big[i]
                        if (i == len(decision_list) - 1):
                            contours_complete.append(connection)
                return contours_complete

    @classmethod
    def ConnectNearbyContours1(self, contours_big, contourDist):
        cnt_size = len(contours_big)
        cnt_dists = np.zeros((cnt_size, cnt_size))

        # Матрица расстояний между контурами
        for row in range(cnt_size):
            M1 = cv2.moments(row)
            cx1 = int(M1['m10'] / M1['m00'])
            cy1 = int(M1['m01'] / M1['m00'])
            
            for col in range(cnt_size):
                if row == col:
                    continue

                if cnt_dists[col, row] != 0:
                    cnt_dists[row, col] = cnt_dists[col, row]
                else:
                    M2 = cv2.moments(col)
                    cx2 = int(M2['m10'] / M2['m00'])
                    cy2 = int(M2['m01'] / M2['m00'])
                    cnt_dists[row, col] = np.linalg.norm(np.array((cx1, cy1))
                                                         - np.array((cx2, cy2)))
        # Объединение близлежащих контуров

        for row in range(cnt_size):
            for col in range(cnt_size):
                if row <= col or cnt_dists[row, col] == 0:
                    continue
                else:
                    pass
