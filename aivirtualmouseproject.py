import sys
import os
import time
import logging
import argparse
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import autopy
import pyautogui

import HandTrackingModule as htm


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

wCam, hCam = 480, 360
frameR = 100
smoothening = 3

click_debounce_sec = 0.12
double_click_threshold = 60
swipe_threshold_px = 120
scroll_threshold_px = 80
cooldown_swipe_sec = 0.7
rest_toggle_hold_sec = 0.35

pTime = 0
plocX, plocY, clocX, clocY = 0, 0, 0, 0
last_click_time, last_action_time = 0.0, 0.0
rest_since, paused = None, False
trail = deque(maxlen=5)


def parse_args():
    parser = argparse.ArgumentParser(description="AI Virtual Mouse")
    parser.add_argument(
        "--source",
        default="0",
        help="0 for webcam or path to a video file",
    )
    parser.add_argument(
        "--record",
        default="on",
        choices=["on", "off"],
        help="Record annotated video (default: on)",
    )
    return parser.parse_args()


def open_capture(source):
    source_str = str(source)

    if source_str != "0":
        cap = cv2.VideoCapture(source_str)
        return cap, source_str

    for camera_index in range(5):
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            return cap, camera_index
        cap.release()

    return None, None


def safe_move_mouse(x, y, wScr, hScr):
    x = max(0, min(wScr - 1, x))
    y = max(0, min(hScr - 1, y))
    try:
        autopy.mouse.move(x, y)
    except Exception as exc:
        logging.debug("Mouse move ignored: %s", exc)


def main():
    global pTime, plocX, plocY, clocX, clocY
    global last_click_time, last_action_time, rest_since, paused

    args = parse_args()
    source = 0 if str(args.source) == "0" else str(args.source)

    cap, active_source = open_capture(source)
    if cap is None or not cap.isOpened():
        if str(args.source) == "0":
            logging.error(
                "Could not open any webcam. Connect or enable a camera, "
                "or run with --source <video_file>."
            )
        else:
            logging.error("Could not open source: %s", source)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)

    detector = htm.handDetector(maxHands=1)
    wScr, hScr = autopy.screen.size()

    out = None
    if args.record == "on":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out = cv2.VideoWriter(
            f"output_{timestamp}.avi",
            fourcc,
            20.0,
            (wCam, hCam),
        )

    logging.info(
        "Virtual Mouse Started on source: %s (ESC or Q to quit)",
        active_source,
    )

    while True:
        success, img = cap.read()
        if not success:
            logging.info("Input stream ended.")
            break

        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img, draw=False)

        hud_y = 30
        cv2.putText(
            img,
            "PAUSED" if paused else "ACTIVE",
            (20, hud_y),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 0, 255) if paused else (0, 255, 0),
            2,
        )

        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            fingers = detector.fingersUp()

            if fingers == [0, 0, 0, 0, 0]:
                if rest_since is None:
                    rest_since = time.time()
                elif (time.time() - rest_since) >= rest_toggle_hold_sec:
                    paused = not paused
                    rest_since = None
                    logging.info("State changed: %s", "Paused" if paused else "Active")
            else:
                rest_since = None

            cv2.rectangle(
                img,
                (frameR, frameR),
                (wCam - frameR, hCam - frameR),
                (255, 0, 255),
                2,
            )

            if not paused:
                if fingers[1] == 1 and sum(fingers) == 1:
                    x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                    y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                    clocX = plocX + (x3 - plocX) / smoothening
                    clocY = plocY + (y3 - plocY) / smoothening
                    safe_move_mouse(wScr - clocX, clocY, wScr, hScr)
                    cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                    plocX, plocY = clocX, clocY

                if fingers[1] == 1 and fingers[2] == 1:
                    length, img, _ = detector.findDistance(8, 12, img)
                    now = time.time()
                    if length < 35 and (now - last_click_time) > click_debounce_sec:
                        pyautogui.click()
                        last_click_time = now
                        logging.info("Left Click")
                    elif (
                        length > double_click_threshold
                        and (now - last_click_time) > click_debounce_sec
                    ):
                        pyautogui.doubleClick()
                        last_click_time = now
                        logging.info("Double Click (V shape)")

                if fingers[0] == 1 and fingers[1] == 1:
                    lengthTI, img, _ = detector.findDistance(4, 8, img)
                    now = time.time()
                    if lengthTI < 30 and (now - last_click_time) > click_debounce_sec:
                        pyautogui.click(button="right")
                        last_click_time = now
                        logging.info("Right Click")

                trail.append((x1, y1, time.time()))
                if len(trail) >= 2:
                    dx = trail[-1][0] - trail[0][0]
                    dy = trail[-1][1] - trail[0][1]
                    now = time.time()

                    if fingers == [1, 1, 1, 1, 1] and (now - last_action_time) > cooldown_swipe_sec:
                        if dx <= -swipe_threshold_px and abs(dy) < swipe_threshold_px / 2:
                            pyautogui.hotkey("ctrl", "a")
                            last_action_time = now
                            logging.info("Swipe Left -> Select All")
                        elif dx >= swipe_threshold_px and abs(dy) < swipe_threshold_px / 2:
                            pyautogui.hotkey("ctrl", "c")
                            last_action_time = now
                            logging.info("Swipe Right -> Copy")

                    if (
                        fingers == [1, 1, 1, 1, 1]
                        or (fingers[1] == 1 and sum(fingers) <= 2)
                    ) and (now - last_action_time) > 0.2:
                        if dy <= -scroll_threshold_px:
                            pyautogui.scroll(60)
                            last_action_time = now
                            logging.info("Scroll Up")
                        elif dy >= scroll_threshold_px:
                            pyautogui.scroll(-60)
                            last_action_time = now
                            logging.info("Scroll Down")

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(
            img,
            f"FPS:{int(fps)}",
            (20, hud_y + 60),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 0, 0),
            2,
        )

        cv2.imshow("AI Virtual Mouse", img)
        if out:
            out.write(img)

        if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
