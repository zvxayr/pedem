import cv2
import pigpio
from RPLCD.pigpio import CharLCD

import config
from canny_edge_detection import detect_edges, slice_image
from foot_measurement import convert_px_to_cm, get_foot_dimensions_px


def process_image(file_path, bounds):
    img = cv2.imread(file_path, 0)
    img = slice_image(img, **bounds)
    edge_img = detect_edges(img)
    return edge_img


def get_foot_size(edge_img):
    foot_dimensions = get_foot_dimensions_px(edge_img)
    cm = convert_px_to_cm(foot_dimensions)
    return cm


if __name__ == "__main__":
    pi = pigpio.pi()
    lcd = CharLCD(pi,
                  pin_rs=15, pin_rw=18, pin_e=16, pins_data=[21, 22, 23, 24],
                  cols=20, rows=4, dotsize=8,
                  charmap='A02',
                  auto_linebreaks=True)

    BUTTON_PIN = 10
    pi.set_mode(BUTTON_PIN, pigpio.INPUT)
    pi.set_pull_up_down(BUTTON_PIN, pigpio.PUD_DOWN)
    pi.set_glitch_filter(BUTTON_PIN, 200)

    file_path = "Samples/S01 27.0.jpg"

    def button_callback(gpio, level, tick):
        if level == pigpio.HIGH:
            lcd.clear()
            lcd.write_string("Processing image...")
            edge_img = process_image(file_path, config.image_bounds)
            cm = get_foot_size(edge_img)
            lcd.clear()
            lcd.write_string(f"Foot size: {cm:.2f} cm")

    cb = pi.callback(BUTTON_PIN, pigpio.RISING_EDGE, button_callback)

    message = input("Press enter to quit\n\n")
    cb.cancel()
    pi.stop()
