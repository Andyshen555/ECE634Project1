import numpy as np
import cv2
from math import log10, sqrt

class hbma:
    def __init__(self, img:np.ndarray, range:int, blockSize:int, level:int) -> None:
        self.range = max(range, 1)
        self.setSize = False
        self.lev = level
        self.blockS = blockSize * 2**(self.lev-1)
        self.height, self.width, _ = img.shape
        self.anchor = img
        self.output = img.copy()

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
        newImg = np.zeros((self.height + 2*self.range, self.width + 2*self.range, 3))
        newImg[self.range:self.range+self.height, self.range:self.range+self.width, :] = img
        return newImg

    def calculate_motion(self, img:np.ndarray) -> np.ndarray:
        img2 = self.pad(img.copy())
        # i, j is the upper left corner of new frame
        # k, l is the upper left corner of anchor frame
        output = None
        for lev in range(self.lev):
            hStep = int(np.ceil(self.height / self.blockS))
            wStep = int(np.ceil(self.width / self.blockS))
            if output is None:
                output = np.zeros((hStep, wStep, 2), np.uint8)
            else:
                output = cv2.resize(output, (wStep, hStep), cv2.INTER_NEAREST)
                output = output * 2
                for k in range(hStep):
                    for l in range(wStep):
                        anchorBlock = self.anchor[k*self.blockS:(k+1)*self.blockS, l*self.blockS:(l+1)*self.blockS, :]
                        highScore = float("inf")
                        for i in range(self.range, min(2*self.range, self.height-output[k, l, 0]-(k+1)*self.blockS)):
                            for j in range(self.range, min(2*self.range, self.width-output[k, l, 1]-(l+1)*self.blockS)):
                                hL = k*self.blockS+output[k, l, 0]+i
                                hH = (k+1)*self.blockS+output[k, l, 0]+i
                                wL = l*self.blockS+output[k, l, 1]+j
                                wH = (l+1)*self.blockS+output[k, l, 1]+j
                                if hL < 0 or wL < 0 or hH > self.height+2*self.range or wH > self.width+2*self.range:
                                    continue
                                newBlock = img2[k*self.blockS+output[k, l, 0]+i:(k+1)*self.blockS+output[k, l, 0]+i, l*self.blockS+output[k, l, 1]+j:(l+1)*self.blockS+output[k, l, 1]+j, :]
                                score = self.getMAD(newBlock, anchorBlock)
                                if score < highScore:
                                    output[k, l, 0] = i
                                    output[k, l, 1] = j
                                    highScore = score
                                    self.output[k*self.blockS:(k+1)*self.blockS, l*self.blockS:(l+1)*self.blockS, :] = newBlock
            self.blockS = self.blockS//2
        return output

    def warp(self, motion:np.ndarray) -> np.ndarray:
        output = self.pad(self.anchor.copy())

        hStep = int(np.ceil(self.height / self.blockS))
        wStep = int(np.ceil(self.width / self.blockS))
        for k in range(hStep):
            for l in range(wStep):
                anchorBlock = self.anchor[k*self.blockS:(k+1)*self.blockS, l*self.blockS:(l+1)*self.blockS, :]
                
                dX = motion[k, l, 0]
                dY = motion[k, l, 1]
                output[k*self.blockS+dX:(k+1)*self.blockS+dX, l*self.blockS+dY:(l+1)*self.blockS+dY, :] = anchorBlock
                
        return output[self.range:-self.range,self.range:-self.range,:]

    def diff(self, anchor:np.ndarray, frame:np.ndarray) -> np.ndarray:
        return np.absolute(anchor - frame)

    def plot_motion(self, motion:np.ndarray) -> np.ndarray:
        h, w, _ = motion.shape
        self.blockS*=2
        output = self.pad(self.anchor.copy())
        pts1 = []
        pts2 = []
        for i in range(h):
            for j in range(w):
                dX = int(motion[i, j, 0]) * 2
                dY = int(motion[i, j, 1]) * 2
                startP = [j*self.blockS+self.blockS//2, i*self.blockS + self.blockS//2]
                endP = [j*self.blockS+self.blockS//2 + dY, i*self.blockS + self.blockS//2 + dX]
                pts1.append(endP)
                pts2.append(startP)
                output = cv2.arrowedLine(output, endP, startP, (255, 0, 0), 1)
        
        self.pts1 = pts1
        self.pts2 = pts2
        return output

    def search(self, img:np.ndarray) -> float:
        motion = self.calculate_motion(img)
        self.motion = motion
        # output = self.warp(motion)
        output = self.output
        diff = self.diff(output, img)
        plotMotion = self.plot_motion(motion)
        # cv2.imshow('Image', img.astype(np.uint8))
        # cv2.imshow('Anchor frame', self.anchor.astype(np.uint8))
        # cv2.imshow('Warpped output', output.astype(np.uint8))
        # cv2.imshow('Diff', diff.astype(np.uint8))
        # cv2.imshow('Motion', plotMotion.astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.imwrite('HBMA-w'+str(self.lev)+'.jpg', output.astype(np.uint8))
        # cv2.imwrite('HBMA-d'+str(self.lev)+'.jpg', diff.astype(np.uint8))
        # cv2.imwrite('HBMA-M'+str(self.lev)+'.jpg', plotMotion.astype(np.uint8))
        return self.PSNR(img, output)

    def getMatchP(self):
        return self.pts1, self.pts2


# img1 = cv2.imread("flower0000.jpg")
# img2 = cv2.imread("flower0062.jpg")
# for l in [2, 3, 4]:
#     matcher = hbma(img1, range=5, blockSize=16, level=l)
#     psnr = matcher.search(img2)
#     print("PSNR for HBMA of range 6, block size 16 and level "+str(l)+" is", psnr)