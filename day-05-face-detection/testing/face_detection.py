import cv2

# Load classifier dan filter
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
filter_img = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)  # pastikan file punya alpha channel

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Resize filter sesuai ukuran wajah
        filter_resized = cv2.resize(filter_img, (w, int(h/3)))
        fw, fh, fc = filter_resized.shape

        # Tempelkan filter di area wajah (misalnya area mata)
        for i in range(fw):
            for j in range(fh):
                if filter_resized[i, j][3] != 0:  # jika bukan pixel transparan
                    frame[y+i, x+j] = filter_resized[i, j][:3]

    cv2.imshow('Face Filter', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
