import cv2
import numpy as np
from matplotlib import pylab as plt

#参照画像(img_ref)と比較画像(img_comp)の読み込み
img_ref  = cv2.imread('./20_ref.png', 1)
img_comp = cv2.imread('./20_comp.png', 1)
temp = img_comp.copy()
#グレースケース変換
gray_img_ref  = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
gray_img_comp = cv2.cvtColor(img_comp, cv2.COLOR_BGR2GRAY)
#参照画像の平滑化 ※変化をつけるためにわざと加えている
gray_img_ref  = cv2.blur(gray_img_ref, (3, 3))
#単純に画像の引き算
img_diff = cv2.absdiff(gray_img_ref, gray_img_comp)
#差分画像の２値化（閾値が50）
ret, img_bin = cv2.threshold(img_diff, 50, 255, 0)
#2値画像に存在する輪郭の座標値を得る
contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#contoursから一個ずつ輪郭を取り出し、輪郭の位置(x,y)とサイズ(width, height)を得る
#サイズが 5x5 以上の輪郭を枠で囲う。
for contour in contours:
    x, y, width, height = cv2.boundingRect(contour)
    if width > 5 or height > 5:
        cv2.rectangle(temp, (x-2, y-2), (x+width+2, y+height+2), (0, 255, 0), 1)
    else:
        continue
#画像表示
cv2.imshow("Original images", np.hstack([img_ref, img_comp]))
cv2.imshow("Processed images", np.hstack([img_diff, img_bin]))
cv2.imshow("Result", temp)
cv2.waitKey(0)
cv2.destroyAllWindows()