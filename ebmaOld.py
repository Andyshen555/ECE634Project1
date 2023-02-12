import numpy as np
import cv2
from math import log10, sqrt

class ebma:
    def __init__(self, img:np.ndarray, range:int, kRaidus:int, half_pel:bool=False) -> None:
        self.half = half_pel
        self.range = range
        self.setSize = False
        self.rad = kRaidus
        self.height, self.width, _ = img.shape
        if half_pel:
            img = cv2.resize(img, (self.width*2, self.height*2), interpolation = cv2.INTER_LINEAR)
        self.anchor = img

    def setAnchor(self, img:np.ndarray) -> None:
        if self.half:
            img = cv2.resize(img, (self.width*2, self.height*2), interpolation = cv2.INTER_LINEAR)
        self.anchor = img
        return

    def PSNR(self, original, compressed):
        mse = np.mean((original - compressed) ** 2)
        if(mse == 0):  # MSE is zero means no noise is present in the signal .
                    # Therefore PSNR have no importance.
            return 100
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))
        return psnr

    def getMAD(self, tBlock, aBlock):
        return np.sum(np.abs(np.subtract(tBlock, aBlock)))/(tBlock.shape[0]*tBlock.shape[1])

    def pad(self, img:np.ndarray) -> np.ndarray:
        newImg = np.zeros((self.height + 2*self.rad, self.width + 2*self.rad, 3))
        newImg[self.rad:self.rad+self.height, self.rad:self.rad+self.width, :] = img
        return newImg

    def calculate_motion(self, img:np.ndarray) -> np.ndarray:
        output = np.zeros((self.height, self.width, 2))
        # i, j is the upper left corner of new frame
        # k, l is the upper left corner of anchor frame
        heightR = min(self.height - 2*self.rad - 1, self.range)
        widthR = min(self.width - 2*self.rad - 1, self.range)
        for k in range(self.height - 2*self.rad - 1):
            print(k)
            for l in range(self.width - 2*self.rad - 1):  
                highScore = -1
                for i in range(heightR):
                    for j in range(widthR):
                        score = self.getMAD(self.anchor[k+self.rad:k+2*self.rad+1, l+self.rad:l+2*self.rad+1], img[i+self.rad:i+2*self.rad+1, j+self.rad:j+2*self.rad+1])
                        if score > highScore:
                            output[k + self.rad, l + self.rad, 0] = i - k
                            output[k + self.rad, l + self.rad, 1] = j - l
        return output

    def warp(self, prevF:np.ndarray, motion:np.ndarray) -> np.ndarray:
        h, w, c = prevF.shape
        output = np.zeros((h, w, c))
        for k in range(self.height - 2*self.rad - 1):
            for l in range(self.width - 2*self.rad - 1):
                dX = motion[k + self.rad, l + self.rad, 0]
                dY = motion[k + self.rad, l + self.rad, 1]
                # print(k + self.rad + dX, l + self.rad + dY)
                output[int(k + self.rad + dX), int(l + self.rad + dY), :] = prevF[int(k + self.rad), int(l + self.rad), :]
        return output

    def diff(self, anchor:np.ndarray, frame:np.ndarray) -> np.ndarray:
        return np.absolute(anchor - frame)

    def search(self, img:np.ndarray) -> None:
        if self.half:
            img = cv2.resize(img, (self.width*2, self.height*2), interpolation = cv2.INTER_LINEAR)
        motion = self.calculate_motion(img)
        output = self.warp(self.anchor, motion)
        diff = self.diff(self.anchor, img)
        cv2.imshow('Image', img.astype(np.uint8))
        cv2.imshow('Warpped output', output.astype(np.uint8))
        cv2.imshow('Diff', diff.astype(np.uint8))
        cv2.waitKey(0)
        print(self.PSNR(img, output))
        return

img1 = cv2.imread("flower0000.jpg")
img2 = cv2.imread("flower0062.jpg")
matcher = ebma(img1, 40, 3)
matcher.search(img2)