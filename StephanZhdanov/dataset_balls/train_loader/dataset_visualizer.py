import cv2

yellow_ball = cv2.imread("yellow_ball/1.jpg")
yellow_ball = cv2.resize(yellow_ball, (124, 124))
yellow_ball_1 = cv2.imread("yellow_ball/2.jpg")
yellow_ball_1 = cv2.resize(yellow_ball_1, (124, 124))
yellow_ball = cv2.hconcat([yellow_ball, yellow_ball_1])

orange_ball = cv2.imread("orange_ball/1655371145.2667649.jpg")
orange_ball = cv2.resize(orange_ball, (124, 124))
orange_ball_1 = cv2.imread("orange_ball/1655371567.1749253.jpg")
orange_ball_1 = cv2.resize(orange_ball_1, (124, 124))
orange_ball = cv2.hconcat([orange_ball, orange_ball_1])



green_ball = cv2.imread("green_ball/1655371285.7846546.jpg")
green_ball = cv2.resize(green_ball, (124, 124))
green_ball_1 = cv2.imread("green_ball/1655371756.677431.jpg")
green_ball_1 = cv2.resize(green_ball_1, (124, 124))
green_ball = cv2.hconcat([green_ball, green_ball_1])



rot_45 = cv2.imread("45/1655720131.7009718.jpg")
rot_45 = cv2.resize(rot_45, (124, 124))
rot_90 = cv2.imread("90/1655720353.1810908.jpg")
rot_90 = cv2.resize(rot_90, (124, 124))

cubes = cv2.hconcat([rot_45, rot_90])


res_img = cv2.vconcat([green_ball, orange_ball, yellow_ball, cubes])

while True:
    cv2.imshow("Dataset", res_img)
    key = cv2.waitKey(0)
    if key == 27:
        break
