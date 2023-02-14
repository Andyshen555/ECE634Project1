import numpy as np
import cv2
from scipy.optimize import least_squares
from ebma import ebma

def random_subset(set1, set2, num):
    idx_list = np.random.permutation(len(set1))
    out1 = []
    out2 = []
    for i in range(num):
        out1.append(set1[idx_list[i]])
        out2.append(set2[idx_list[i]])
    return out1, out2

def computeH(img1, img2):
    # Compute H according to lecture note 5
    # img1 maps to img2
    x = np.zeros((len(img1) * 2, 8))
    y = np.zeros((len(img1) * 2, 1))
    for i in range(len(img1)):
        x[2 * i] = [img1[i][0], img1[i][1], 1, 0, 0, 0, -img1[i][0] * img2[i][0], -img1[i][1] * img2[i][0]]
        x[2 * i + 1] = [0, 0, 0, img1[i][0], img1[i][1], 1, -img1[i][0] * img2[i][1], -img1[i][1] * img2[i][1]]
        y[2 * i] = img2[i][0]
        y[2 * i + 1] = img2[i][1]

    h = np.append(np.dot(np.linalg.pinv(x), y), 1)
    return h.reshape((3, 3))

def transform(pt2, H):
    pt = pt2[0:2]
    pt.append(1)
    new_pt = np.dot(H, pt)
    return new_pt / new_pt[2]

def find_inlier(pts1, pts2, H, delta):
    out1 = []
    out2 = []
    for i in range(len(pts1)):
        if np.linalg.norm(pts2[i] - transform(pts1[i], H)[0:2]) < delta:
            out1.append(pts1[i])
            out2.append(pts2[i])
    return out1, out2

# def find_inlier(pts1, pts2, H, delta):
#     out1 = []
#     out2 = []
#     for i in range(len(pts1)):
#         if np.linalg.norm(pts2[i] - transform(pts1[i], H)[0:2]) >= delta:
#             out1.append(pts1[i])
#             out2.append(pts2[i])
#     return out1, out2

def ransac(pts1, pts2, epsilon, n, p, delta):
    N = int(np.log(1-p) / np.log(1 - (1 - epsilon) ** n))
    save1 = []
    save2 = []
    for i in range(100):
        sub1, sub2 = random_subset(pts1, pts2, 4)
        H = computeH(sub1, sub2)
        in1, in2 = find_inlier(pts1, pts2, H, delta)
        if len(in1) > len(save1):
            save1 = in1
            save2 = in2
    return save1, save2

def linearH(pts1, pts2):
    lens = len(pts1)
    if lens % 2 != 0:
        pts1.append ([0, 0])
        pts2.append ([0, 0])

    A = np.zeros((2 * lens, 8))
    b = np.zeros((2 * lens, 1))

    for i in range (lens):
        pt1 = np.array([pts1[i][0], pts1[i][1], 1])
        pt2 = np.array([pts2[i][0], pts2[i][1], 1])
        A[2 * i] = [0, 0, 0, -pt1[0], - pt1[1], -1, pt1[0], pt2[1] * pt1[1]]
        b[2 * i] = -pt2[1] * pt1[2]
        A[2 * i + 1] = [pt1[0], pt1[1], 1, 0, 0, 0, - pt2[0] * pt1[0], -pt2[0] * pt1[1]]
        b[2 * i + 1] = pt2[0] * pt1[2]

    h = np.dot(np.linalg.pinv(A), b)
    h = np.append(h, 1)

    return h.reshape((3, 3))

def canvas_size(img , H):
    (h, w, _) = img.shape
    b = np.zeros ((4, 3))
    b[0] = np.dot(H, np.array([0, 0, 1])) / np.dot(H, np.array([0, 0, 1]))[2]
    b[1] = np.dot(H, np.array([h, 0, 1])) / np.dot(H, np.array([h, 0, 1]))[2]
    b[2] = np.dot(H, np.array([0, w, 1])) / np.dot(H, np.array([0, w, 1]))[2]
    b[3] = np.dot(H, np.array([h, w, 1])) / np.dot(H, np.array([w, h, 1]))[2]

    return b

def empty_canvas(img0, img1, img3, img4, H02, H12, H32, H42):
    b1 = canvas_size(img0, H02)
    b2 = canvas_size(img1, H12)
    b3 = canvas_size(img3, H32)
    b4 = canvas_size(img4, H42)

    x_min, y_min, _ = np.amin(np.amin([b1, b2, b3, b4], 0), 0)
    x_max, y_max, _ = np.amax(np.amax([b1, b2, b3, b4], 0), 0)
    w = int(round(x_max)) - int(round(x_min))
    h = int(round(y_max)) - int(round(y_min))
    canvas = np.zeros((h, w, 3))

    return canvas, x_min, y_min

def transform2(img, img2, h, x_min, y_min):
    # img2 to img1 map
    # h is img2 to img1 map
    h = np.linalg.pinv(h) / np.linalg.pinv(h)[2][2]
    img1 = img.copy()
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            newCor = np.dot(h, np.array([j - y_min, i - x_min, 1]))
            l = round(newCor[0]/newCor[2])
            k = round(newCor[1]/newCor[2])
            if k < img2.shape[0] and k >= 0 and l < img2.shape[1] and l >= 0:
                img1[i][j] = img2[k][l]
    return img1

def show_liers(img1, img2, inlier_pt1, inlier_pt2, name):
    img = np.hstack((img1, img2))
    h, w, d = img1.shape

    for i in range(len(inlier_pt1)):
        pt1 = [int(inlier_pt1[i][0]), int(inlier_pt1[i][1])]
        pt2 = [int(inlier_pt2[i][0] + w), int(inlier_pt2[i][1])]
        cv2.line(img, tuple(pt1), tuple(pt2), (0, 255, 255), 1)
        cv2.circle(img, tuple(pt1), 4, (0, 255, 255), 2)
        cv2.circle(img, tuple(pt2), 4, (0, 255, 255), 2)
    cv2.imwrite(name,img)
    return

def find_M(matcher, knr1, knr2, num):
  # Initialize lists
  list_kp1 = []
  list_kp2 = []
  cnt = 0

  # For each match...
  for mat in matcher:
      if cnt >= num:
          break
      # Get the matching keypoints for each of the images
      img1_idx = mat.queryIdx
      img2_idx = mat.trainIdx

      # x - columns
      # y - rows
      # Get the coordinates
      [x1, y1] = knr1[img1_idx].pt
      [x2, y2] = knr2[img2_idx].pt

      # Append to each list
      list_kp1.append([x1, y1])
      list_kp2.append([x2, y2])
      cnt = cnt + 1

  return list_kp1, list_kp2

def nonLinearH(pts1, pts2, H):
    
    def func(h):
        loss = []

        for i in range(len(pts1)):
            x = pts1[i][0] * h[0] + pts1[i][1] * h[1] + h[2] / (pts1[i][0] * h[6] + pts1[i][7] * h[4] + 1)
            loss.append(pts2[i][0] - x)
            y = pts1[i][0] * h[3] + pts1[i][1] * h[1] + h[5] / (pts1[i][0] * h[6] + pts1[i][7] * h[4] + 1)
            loss.append(pts2[i][1] - y)
            
        return np.asarray(loss)

    sol = least_squares(func, H.squeeze(), method='lm')
    output = sol.x
    output.append(1)
    return output.reshape((3, 3))

# read image
img0 = cv2.imread("akiyo0000.jpg")
img1 = cv2.imread("akiyo0031.jpg")
matcher = ebma(img0, range=10, blockSize=16, half_pel=False)
motion = matcher.search(img1)
pts1, pts2 = matcher.getMatchP()
print("**********Read Image Finished**********")


# RANSAC paramters
epsilon = 0.1
delta = 8
p = 0.99
n = 4

in0, in01 = ransac(pts2, pts1, n, epsilon, p, delta)
print("**********RANSAC Finished**********")

H_01 = linearH(in0, in01)
print(H_01)
print("**********H Compute Finished**********")

canvas = transform2(img1, img0, H_01, 0, 0)
print("**********Linear Output Finished**********")

print(matcher.PSNR(img0, canvas))
diff = matcher.diff(canvas, img0)
cv2.imwrite('Diff.jpg', diff.astype(np.uint8))

cv2.imwrite('output.jpg', canvas)

# H_01 = nonLinearH(in0, in01, H_01)
print("**********NonLinear Finished**********")

# canvas = transform2(img1, img0, H_01, 0, 0)

# cv2.imwrite('output.jpg', canvas)
print("**********Output Finished**********")