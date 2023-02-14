import numpy as np
import cv2
from math import log10, sqrt

class ebma:
    def __init__(self, img:np.ndarray, range:int, blockSize:int, half_pel:bool=False) -> None:
        self.half = half_pel
        self.range = max(range, 1)
        self.height, self.width, _ = img.shape
        self.blockS = blockSize
        if half_pel:
            self.width *=2
            self.height *= 2
            img = cv2.resize(img, (self.width, self.height), interpolation = cv2.INTER_LINEAR)
            self.blockS *=2
            self.range *=2
        self.anchor = img # this is acutally the frame
        self.output = img.copy()

    def setAnchorFrame(self, img:np.ndarray) -> None:
        if self.half:
            img = cv2.resize(img, (self.width*2, self.height*2), interpolation = cv2.INTER_LINEAR)
        self.anchor = img
        return

    def PSNR(self, original:np.ndarray, compressed:np.ndarray) -> float:
        mse = np.mean((original - compressed) ** 2)
        if(mse == 0):  # MSE is zero means no noise is present in the signal.
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
        output = np.zeros((hStep, wStep, 2), np.uint8)
        for k in range(hStep):
            for l in range(wStep):
                anchorBlock = self.anchor[k*self.blockS:(k+1)*self.blockS, l*self.blockS:(l+1)*self.blockS, :]
                highScore = float("inf")
                for i in range(self.range, 2*self.range):
                    for j in range(self.range, 2*self.range):
                        newBlock = img2[k*self.blockS+i:(k+1)*self.blockS+i, l*self.blockS+j:(l+1)*self.blockS+j, :]
                        score = self.getMAD(newBlock, anchorBlock)
                        if score < highScore:
                            output[k, l, 0] = i
                            output[k, l, 1] = j
                            highScore = score
                            self.output[k*self.blockS:(k+1)*self.blockS, l*self.blockS:(l+1)*self.blockS, :] = newBlock
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

    def diff(self, wrapped:np.ndarray, frame:np.ndarray) -> np.ndarray:
        return np.absolute(wrapped - frame)

    def plot_motion(self, motion:np.ndarray) -> np.ndarray:
        h, w, _ = motion.shape
        output = self.pad(self.anchor.copy())
        pts1 = []
        pts2 = []
        for i in range(h):
            for j in range(w):
                dX = int(motion[i, j, 0])
                dY = int(motion[i, j, 1])
                startP = [j*self.blockS+self.blockS//2, i*self.blockS + self.blockS//2]
                endP = [j*self.blockS+self.blockS//2 + dY, i*self.blockS + self.blockS//2 + dX]
                pts1.append(endP)
                pts2.append(startP)
                output = cv2.arrowedLine(output, endP, startP, (255, 0, 0), 1)
        self.pts1 = pts1
        self.pts2 = pts2
        return output

    def search(self, img:np.ndarray) -> float:
        if self.half:
            img = cv2.resize(img, (self.width, self.height), interpolation = cv2.INTER_LINEAR)
        motion = self.calculate_motion(img)
        self.motion = motion
        # output = self.warp(motion)
        output = self.output
        diff = self.diff(output, img)
        plotMotion = self.plot_motion(motion)
        # if self.half:
        #     output = cv2.resize(output, (self.width//2, self.height//2), interpolation = cv2.INTER_LINEAR)
        #     diff = cv2.resize(diff, (self.width//2, self.height//2), interpolation = cv2.INTER_LINEAR)
        #     plotMotion = cv2.resize(plotMotion, (self.width//2, self.height//2), interpolation = cv2.INTER_LINEAR)
        # cv2.imwrite('EBMA-H-w'+str(self.range)+'-'+str(self.blockS)+'.jpg', output.astype(np.uint8))
        # cv2.imwrite('EBMA-H-d'+str(self.range)+'-'+str(self.blockS)+'.jpg', diff.astype(np.uint8))
        # cv2.imwrite('EBMA-H-M'+str(self.range)+'-'+str(self.blockS)+'.jpg', plotMotion.astype(np.uint8))
        return self.PSNR(img, output)

    def getMatchP(self):
        return self.pts1, self.pts2


# img1 = cv2.imread("flower0000.jpg")
# img2 = cv2.imread("flower0062.jpg")
# for r in [3, 6, 9]:
#     for blk in [4, 8, 16]:
#         matcher = ebma(img1, range=r, blockSize=blk, half_pel=True)
#         psnr = matcher.search(img2)
#         print("PSNR for half pel EBMA of range "+str(r)+" and block size "+str(blk)+" is", psnr)