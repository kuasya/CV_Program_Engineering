import cv2
import numpy as np


def main():
    video_path = 'mona-lisa-blur-extra-credit.avi'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Ошибка: Видео не найдено.")
        return

    ret, first_frame = cap.read()
    if not ret: return

    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    print("Выделите объект и нажмите ENTER/SPACE.")
    bbox = cv2.selectROI("Select Target", first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Target")

    if bbox == (0, 0, 0, 0): return
    x, y, w, h = map(int, bbox)

    target_gray = first_gray[y:y + h, x:x + w]
    target_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    sift = cv2.SIFT_create(contrastThreshold=0.02)
    kp_ref, des_ref = sift.detectAndCompute(target_gray, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    feature_params = dict(maxCorners=50, qualityLevel=0.05, minDistance=15, blockSize=7)
    ref_pts_local = cv2.goodFeaturesToTrack(target_gray, mask=None, **feature_params)

    if ref_pts_local is None:
        print("Не найдено текстур.")
        return

    p0 = ref_pts_local + np.float32([x, y])
    active_ref_pts = ref_pts_local.copy()
    old_gray = first_gray.copy()

    lk_params = dict(winSize=(45, 45), maxLevel=5, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    tracking_mode = "FLOW"

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        target_found = False

        if tracking_mode == "FLOW":
            if len(p0) > 0:
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

                good_new = p1[st == 1]
                good_ref = active_ref_pts[st == 1]

                if len(good_new) > 10:
                    # Вычисляем гомографию
                    matrix, mask = cv2.findHomography(good_ref, good_new, cv2.RANSAC, 3.0)

                    if matrix is not None:
                        transformed_corners = cv2.perspectiveTransform(target_corners, matrix)
                        if cv2.isContourConvex(np.int32(transformed_corners)):
                            cv2.polylines(frame, [np.int32(transformed_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)
                            for pt in good_new:
                                cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)

                            text_x, text_y = int(transformed_corners[0][0][0]), int(transformed_corners[0][0][1])
                            cv2.putText(frame, 'Mona Lisa', (text_x, text_y - 10), cv2.FONT_HERSHEY_DUPLEX, 1,
                                        (0, 0, 0), 5)
                            cv2.putText(frame, 'Mona Lisa', (text_x, text_y - 10), cv2.FONT_HERSHEY_DUPLEX, 1,
                                        (255, 255, 255), 2)

                            # Обновляем кадр и точки
                            old_gray = frame_gray.copy()
                            p0 = good_new.reshape(-1, 1, 2)
                            active_ref_pts = good_ref.reshape(-1, 1, 2)
                            target_found = True

            if not target_found:
                tracking_mode = "RECOVERY"

        if tracking_mode == "RECOVERY":
            cv2.putText(frame, 'Recovering...', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

            kp_frame, des_frame = sift.detectAndCompute(frame_gray, None)
            if des_frame is not None and len(des_frame) > 2:
                matches = flann.knnMatch(des_ref, des_frame, k=2)
                good_matches = [m for m_n in matches if len(m_n) == 2 for m, n in [m_n] if
                                m.distance < 0.75 * n.distance]

                if len(good_matches) > 15:
                    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    if matrix is not None:
                        transformed_corners = cv2.perspectiveTransform(target_corners, matrix)
                        if cv2.isContourConvex(np.int32(transformed_corners)):

                            restored_p0 = cv2.perspectiveTransform(ref_pts_local, matrix)

                            p0 = restored_p0
                            active_ref_pts = ref_pts_local.copy()
                            old_gray = frame_gray.copy()
                            tracking_mode = "FLOW"

                            cv2.polylines(frame, [np.int32(transformed_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()