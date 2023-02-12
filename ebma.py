import numpy as np
import cv2
from math import log10, sqrt

class ebma:
    def __init__(self, img:np.ndarray, range:int, blockSize:int, half_pel:bool=False) -> None:
        self.half = half_pel
        self.range = max(range, 1)
        self.setSize = False
        self.blockS = blockSize
        self.height, self.width, _ = img.shape
        if half_pel:
            img = cv2.resize(img, (self.width*2, self.height*2), interpolation = cv2.INTER_LINEAR)
            self.blockS *=2
        self.anchor = img

    def setAnchorFrame(self, img:np.ndarray) -> None:
        if self.half:
            img = cv2.resize(img, (self.width*2, self.height*2), interpolation = cv2.INTER_LINEAR)
        self.anchor = img
        return

    def PSNR(self, original:np.ndarray, compressed:np.ndarray) -> float:
        mse = np.mean((original - compressed) ** 2)
        if(mse == 0):  # MSE is zero means no noise is present in the signal .
                    # Therefore PSNR have no importance.
            return 100
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))
        return psnr

    def getMAD(self, tBlock:np.ndarray, aBlock:np.ndarray) -> float:
        return np.mean(np.abs(np.subtract(tBlock, aBlock)))

    def pad(self, img:np.ndarray) -> np.ndarray:
        newImg = np.zeros((self.height + 2*self.range, self.width + 2*self.range, 3))
        newImg[self.range:self.range+self.height, self.range:self.range+self.width, :] = img
        return newImg

    def calculate_motion(self, img:np.ndarray) -> np.ndarray:
        img2 = self.pad(img.copy())
        hStep = int(np.ceil(self.height / self.blockS))
        wStep = int(np.ceil(self.width / self.blockS))
        output = np.zeros((hStep, wStep, 2))
        for k in range(hStep):
            for l in range(wStep):
                anchorBlock = self.anchor[k*self.blockS:(k+1)*self.blockS, l*self.blockS:(l+1)*self.blockS, :]
                highScore = float("inf")
                for i in range(self.range, 2*self.range):
                    for j in range(self.range, 2*self.range):
                        newBlock = img2[k*self.blockS+i:(k+1)*self.blockS+i, l*self.blockS+j:(l+1)*self.blockS+j, :]
                        score = self.getMAD(anchorBlock, newBlock)
                        if score < highScore:
                            output[k, l, 0] = i
                            output[k, l, 1] = j
                            highScore = score
        return output

    def warp(self, motion:np.ndarray) -> np.ndarray:
        output = self.pad(self.anchor.copy())

        hStep = int(np.ceil(self.height / self.blockS))
        wStep = int(np.ceil(self.width / self.blockS))
        for k in range(hStep):
            for l in range(wStep):
                anchorBlock = self.anchor[k*self.blockS:(k+1)*self.blockS, l*self.blockS:(l+1)*self.blockS, :]
                
                dX = int(motion[k, l, 0])
                dY = int(motion[k, l, 1])
                output[k*self.blockS+dX:(k+1)*self.blockS+dX, l*self.blockS+dY:(l+1)*self.blockS+dY, :] = anchorBlock
                
        return output[self.range:-self.range,self.range:-self.range,:]

    def diff(self, wrapped:np.ndarray, frame:np.ndarray) -> np.ndarray:
        return np.absolute(wrapped - frame)

    def plot_motion(self, motion:np.ndarray) -> np.ndarray:
        h, w, _ = motion.shape
        output = self.pad(self.anchor.copy())
        for i in range(h):
            for j in range(w):
                dX = int(motion[i, j, 0])
                dY = int(motion[i, j, 1])
                startP = (j*self.blockS+self.blockS//2, i*self.blockS + self.blockS//2)
                endP = (j*self.blockS+self.blockS//2 + dY, i*self.blockS + self.blockS//2 + dX)
                output = cv2.arrowedLine(output, startP, endP, (255, 0, 0), 1)
        return output[self.range:-self.range,self.range:-self.range,:]

    def search(self, img:np.ndarray) -> None:
        if self.half:
            img = cv2.resize(img, (self.width*2, self.height*2), interpolation = cv2.INTER_LINEAR)
        motion = self.calculate_motion(img)
        output = self.warp(motion)
        # print(output[:,:, 0])
        # print(output[:,:, 1])
        diff = self.diff(output, img)
        plotMotion = self.plot_motion(motion)
        print(self.PSNR(img, output))
        cv2.imshow('Anchor frame', self.anchor.astype(np.uint8))
        cv2.imshow('Image', img.astype(np.uint8))
        cv2.imshow('Warpped output', output.astype(np.uint8))
        cv2.imshow('Diff', diff.astype(np.uint8))
        cv2.imshow('Motion', plotMotion.astype(np.uint8))
        cv2.waitKey(0)
        return

img1 = cv2.imread("flower0000.jpg")
img2 = cv2.imread("flower0062.jpg")
matcher = ebma(img1, range=50, blockSize=4, half_pel=False)
matcher.search(img2)