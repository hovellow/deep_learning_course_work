import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
# использовали медиапайп в котором есть готовая модель для обнаружения лица человека

def test_static_imgs(img_paths, output_path):
    with mp_face_detection.FaceDetection(
            min_detection_confidence=0.5) as face_detection:
        for idx, img_path in enumerate(img_paths):
            image = cv2.imread(img_path)
            W = image.get(3)
            H = image.get(4)
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.detections:
                continue
            annotated_image = image.copy()
            for detection in results.detections:
                bb = detection.location_data.relative_bounding_box
                x = bb.xmin * W
                y = bb.ymin * H
                w = bb.width * W
                h = bb.height * H
                x, y, w, h = int(x - 0.1 * w), int(y - 0.1 * h), int(w * 1.2), int(h * 1.2)
                kernel = 50
                image[y:y + h, x:x + w] = cv2.blur(image[y:y + h, x:x + w], (kernel, kernel))
            cv2.imwrite(output_path + str(idx) + '.png', annotated_image)

#обработка видео
def infer_video(cap):
    W = cap.get(3)
    H = cap.get(4)

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # получаем для конкрентного изображения, возрвращает список найденых объектов,
            # каждый из которых имеет 16 полей, первые четыре являются координатами рамки вокруг лица
            # остальные 12 содержат координаты ключевых точек лица(нос, глаза, уши)
            results = face_detection.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # если найдены лица, тогда ля каждой ограничевающей рамки мы применяем функцию blur из библиотеки cv2
            if results.detections:
                for detection in results.detections:
                    bb = detection.location_data.relative_bounding_box
                    x = abs(bb.xmin * W)
                    y = abs(bb.ymin * H)
                    w = abs(bb.width * W)
                    h = abs(bb.height * H)
                    x, y, w, h = int(x - 0.1 * w), int(y - 0.1 * h), int(w * 1.2), int(h * 1.2)
                    x = 0 if x < 0 else x
                    y = 0 if y < 0 else y

                    kernel = 50 # сила размытия
                    # заменяем данный участок картинки его заблюренным вариантом
                    image[y:y + h, x:x + w] = cv2.blur(image[y:y + h, x:x + w], (kernel, kernel))

            cv2.imshow('Anon Face Detection', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()


def test_video(video_path):
    cap = cv2.VideoCapture(video_path)
    infer_video(cap)


def test_webcam(ip_cam_link=None):
    cap = cv2.VideoCapture(0)
    if ip_cam_link:
        cap.open(ip_cam_link)

    infer_video(cap)


if __name__ == '__main__':
    test_webcam()