import numpy as np
import cv2
from math import log10, sqrt

class ebma:
    def __init__(self, img:np.ndarray, range:int, blockSize:int, level:int) -> None:
        self.range = max(range, 1)
        self.setSize = False
        self.blockS = blockSize
        self.height, self.width, _ = img.shape
        self.anchor = img
        self.lev = level

    def setAnchorFrame(self, img:np.ndarray) -> None:
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
        return np.mean(np.abs(np.subtract(tBlock, aBlock)))

    def pad(self, img:np.ndarray) -> np.ndarray:
        newImg = np.zeros((self.height + 2*self.rad, self.width + 2*self.rad, 3))
        newImg[self.rad:self.rad+self.height, self.rad:self.rad+self.width, :] = img
        return newImg

    def calculate_motion(self, img:np.ndarray) -> np.ndarray:
        # i, j is the upper left corner of new frame
        # k, l is the upper left corner of anchor frame
        for lev in range(self.lev):
            hStep = int(np.ceil(self.height / self.blockS))
            wStep = int(np.ceil(self.width / self.blockS))
            output = np.zeros((hStep, wStep, 2))
            for k in range(hStep):
                for l in range(wStep):
                    anchorBlock = self.anchor[k*self.blockS:(k+1)*self.blockS, l*self.blockS:(l+1)*self.blockS, :]
                    highScore = float("inf")
                    for i in range(max(0, k-self.range), min(k+self.range, hStep)):
                        for j in range(max(0, l-self.range), min(l+self.range, wStep)):
                            newBlock = img[i*self.blockS:(i+1)*self.blockS, j*self.blockS:(j+1)*self.blockS, :]
                            score = self.getMAD(anchorBlock, newBlock)
                            if score < highScore:
                                output[k, l, 0] = i
                                output[k, l, 1] = j
                                highScore = score
        return output

    def warp(self, prevF:np.ndarray, motion:np.ndarray) -> np.ndarray:
        h, w, c = prevF.shape
        output = self.anchor.copy()

        hStep = int(np.ceil(self.height / self.blockS))
        wStep = int(np.ceil(self.width / self.blockS))
        for k in range(hStep):
            for l in range(wStep):
                anchorBlock = self.anchor[k*self.blockS:(k+1)*self.blockS, l*self.blockS:(l+1)*self.blockS, :]
                
                dX = int(motion[k, l, 0])
                dY = int(motion[k, l, 1])
                # print(dX, dY, k, l)
                output[(dX)*self.blockS:(dX+1)*self.blockS, (dY)*self.blockS:(dY+1)*self.blockS, :] = anchorBlock
                
                # cv2.imshow('test', output.astype(np.uint8))
                # cv2.waitKey(0)
        return output

    def diff(self, anchor:np.ndarray, frame:np.ndarray) -> np.ndarray:
        return np.absolute(anchor - frame)

    def search(self, img:np.ndarray) -> None:
        motion = self.calculate_motion(img)
        output = self.warp(self.anchor, motion)
        # print(output[:,:, 0])
        # print(output[:,:, 1])
        diff = self.diff(self.anchor, img)
        cv2.imshow('Image', img.astype(np.uint8))
        cv2.imshow('Anchor frame', self.anchor.astype(np.uint8))
        cv2.imshow('Warpped output', output.astype(np.uint8))
        cv2.imshow('Diff', diff.astype(np.uint8))
        cv2.waitKey(0)
        print(self.PSNR(img, output))
        return

img1 = cv2.imread("flower0000.jpg")
img2 = cv2.imread("flower0062.jpg")
matcher = ebma(img1, range=3, blockSize=16, half_pel=False)
matcher.search(img2)