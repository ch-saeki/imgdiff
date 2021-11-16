import cv2

path1 = '/？？？/sample1-1.png'
path2 = '/？？？/sample2-2.png'
img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

clahe = cv2.createCLAHE(clipLimit=30.0, tileGridSize=(10,10))
img1 = clahe.apply(img1)
img2 = clahe.apply(img2)

img1 = cv2.GaussianBlur(img1,(13,13),0)
img2 = cv2.GaussianBlur(img2,(13,13),0)

diff = cv2.absdiff(img1, img2)
ret, diff = cv2.threshold(diff, 60, 255, cv2.THRESH_BINARY)
diff = cv2.GaussianBlur(diff,(11,11),0)

img2 = cv2.imread(path2)
image, contours, hierarchy = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if w > 15 and h > 15:
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 5)

cv2.imshow('result image', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# https://sosotata.com/spot7differences/