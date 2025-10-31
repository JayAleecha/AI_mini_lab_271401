import socket
import cv2
import numpy as np
from threading import Thread
import time

# =======================
# CONFIG
# =======================
IP_ROBOT = "192.168.1.6"
PORT = 6601
MIN_AREA = 4500

# =======================
# SOCKET CONNECTION
# =======================
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (IP_ROBOT, PORT)
print('connecting to %s port %s' % server_address)
sock.connect(server_address)

pos_frame = []
status = 'wait'
current_color = None  # สีที่ต้องการให้ Vision หาปัจจุบัน

# =======================
# MG400 THREAD
# =======================
class Mg400(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.daemon = True


        sock.send('hi'.encode())
        print('hi')


        # เรียกใช้ฟังก์ชัน input
        self.pickup_list = self.ask_user_input()
        self.picked_count = 0


        print("\n List to pick:")
        for i, color in enumerate(self.pickup_list, 1):
            print(f"  Box {i}: color {color}")
        print("====================================\n")


        # เริ่ม Thread
        self.start()


    def ask_user_input(self):
        while True:
            try:
                n = int(input("How many box: "))
                if n > 0:
                    break
                else:
                    print("Please put number more than 0")
            except ValueError:
                print("Please put a valid number")


        pickup_list = []
        available_colors = ['Red', 'Green', 'Blue', 'Yellow']
        for i in range(n):
            while True:
                color = input(f"Box number {i+1} what color (Red/Green/Blue/Yellow): ").capitalize()
                if color in available_colors:
                    pickup_list.append(color)
                    break
                else:
                    print("Please select color from Red, Green, Blue, or Yellow only")
        return pickup_list


    def run(self):
        global status, pos_frame, current_color


        while True:
            # ถ้าหยิบครบทุกกล่องแล้ว → ถาม input ใหม่
            if self.picked_count >= len(self.pickup_list):
                print("\n All boxes picked.")
                print("====================================")
                # เริ่มรอบใหม่
                self.pickup_list = self.ask_user_input()
                self.picked_count = 0
                print("\n List to pick:")
                for i, color in enumerate(self.pickup_list, 1):
                    print(f"  Box number {i}: color {color}")
                print("====================================\n")
                time.sleep(1)
                continue


            if status == 'wait':
                data = sock.recv(50)


                if data == b'start':
                    # ตั้งค่าสีที่กำลังจะหา
                    current_color = self.pickup_list[self.picked_count]
                    print(f"\n Finding box color {current_color}")
                    status = 'find'
                    time.sleep(1)


                if data == b'pos?':
                    status = 'find_pos'
                    time.sleep(1)


            if status == 'find':
                while True:
                    if pos_frame:
                        msg = f'found'
                        sock.send(msg.encode())
                        print(f'found (box number {self.picked_count + 1} / {len(self.pickup_list)})')
                        status = 'wait'
                        break
                    else:
                        print(f"Not Found!! (Finding box color {current_color})")
                    time.sleep(0.2)


            if status == 'find_pos':
                if pos_frame:
                    print(f'Found:{pos_frame}')
                    x, y, r = to_pos_robot(pos_frame)
                    msg = f'{x:.2f},{y:.2f},{r:.2f}'
                    sock.send(msg.encode())
                    print(msg)


                    self.picked_count += 1
                    print(f" Picked box number {self.picked_count}/{len(self.pickup_list)} finish\n")


                    status = 'wait'
                    current_color = None  # เคลียร์สี
                else:
                    print('Not found')
                    msg = f'finish'
                    sock.send(msg.encode())


            time.sleep(0.1)


# =======================
# TRANSFORM MATRIX
# =======================
camera_points = np.float32([[118, 138], [518, 138], [118, 402], [518, 402]])
world_points = np.float32([
    [257.67, -84.57],
    [263.11, 102.50],
    [380.00, -82.58],
    [380.86, 96.27]
])

matrix = cv2.getPerspectiveTransform(camera_points, world_points)

def to_pos_robot(box1):
    camera_x, camera_y, r = box1[0], box1[1], box1[2]
    camera_coord = np.float32([[camera_x, camera_y]])
    robot_coord = cv2.perspectiveTransform(camera_coord.reshape(-1, 1, 2), matrix)
    robotx, roboty = robot_coord[0][0][0], robot_coord[0][0][1]
    return robotx, roboty, 90 - r

# =======================
# VISION THREAD
# =======================
cap = cv2.VideoCapture(0)

class VisionProcessing(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.daemon = True
        self.start()

    def run(self):
        global status, pos_frame, current_color

        # HSV bounds for color detection
        R_bound = (np.array([0, 80, 0]), np.array([15, 200, 255]))
        G_bound = (np.array([45, 38, 80]), np.array([76, 255, 255]))
        B_bound = (np.array([83, 36, 155]), np.array([134, 255, 255]))
        Y_bound = (np.array([10, 50, 180]), np.array([40, 220, 255]))

        color_bounds = {
            'Red': R_bound,
            'Green': G_bound,
            'Blue': B_bound,
            'Yellow': Y_bound
        }

        kernel = np.ones((5, 5), np.uint8)

        while True:
            time.sleep(0.1)
            ret, frame = cap.read()
            if not ret:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            pos_frame_0 = []


            # ตรวจจับเฉพาะสีที่ผู้ใช้เลือกตอนนี้
            if current_color and current_color in color_bounds:
                lower_bound, upper_bound = color_bounds[current_color]
                mask = cv2.inRange(hsv, lower_bound, upper_bound)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:
                        rect = cv2.minAreaRect(contour)
                        box = cv2.boxPoints(rect)
                        box = box.astype(np.int32)
                        cv2.drawContours(frame, [box], 0, (0, 255, 255), 2)


                        angle = rect[-1]
                        if angle < -45:
                            angle = 90 + angle
                        angle = round(angle, 0)


                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            pos_frame_0.append([cX, cY, angle])


                            cv2.circle(frame, (cX, cY), 3, (0, 0, 255), -1)
                            cv2.putText(frame, f"{cX,cY} {angle:.0f}",
                                        (cX - 50, cY - 30), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 255, 255), 2)

            if pos_frame_0:
                pos_frame = pos_frame_0[0]
            else:
                pos_frame = False

            cv2.circle(frame, (118, 138), 3, (0, 0, 255), -1)
            cv2.circle(frame, (518, 138), 3, (0, 0, 255), -1)
            cv2.circle(frame, (118, 402), 3, (0, 0, 255), -1)
            cv2.circle(frame, (518, 402), 3, (0, 0, 255), -1)


            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        cap.release()
        cv2.destroyAllWindows()

# =======================
# MAIN
# =======================
def main():
    mg400_thread = Mg400()
    vision_thread = VisionProcessing()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    main()
