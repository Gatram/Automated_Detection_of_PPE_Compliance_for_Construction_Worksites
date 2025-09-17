from flask import Flask, request, render_template
import cv2
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO("models/best_030124.pt")

# SMTP Email Configuration
SMTP_SERVER = "smtp.gmail.com"  # Change based on your email provider
SMTP_PORT = 587
EMAIL_ADDRESS = "gatramjyotsna1@gmail.com"
EMAIL_PASSWORD = "edmz zyxu luoq fcdh"
SENDER_MAIL = "21jr1a4417@gmail.com"

def send_alert(missing_gear):
    subject = "Safety Alert: Missing Safety Gear Detected"
    body = f"The following safety gear is missing: {', '.join(missing_gear)}. Please watch it carefully!"
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = SENDER_MAIL  # Change as needed
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, SENDER_MAIL, msg.as_string())
        server.quit()
        print("Email alert sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")


@app.route('/', methods=['GET', 'POST'])
def index():
    missing_gear = []
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = os.path.join("static/uploads", file.filename)
            file.save(file_path)

            # Perform prediction
            results = model.predict(source=file_path, conf=0.1)

            for result in results:
                detected_objects = [model.names[int(box.cls)] for box in result.boxes]
                for item in detected_objects:
                    if item in ["no glove", "no helmet", "no vest", "no boots"]:
                        missing_gear.append(item[3:])

            missing_gear = list(set(missing_gear))  # Remove duplicates

            if missing_gear:
                send_alert(missing_gear)

            return render_template('index09.html', missing_gear=missing_gear)

    return render_template('index09.html', missing_gear=None)


if __name__ == '__main__':
    app.run(debug=True)