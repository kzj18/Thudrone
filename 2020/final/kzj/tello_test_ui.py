#!/usr/bin/python
#-*- encoding: utf8 -*-

from PySide2.QtWidgets import QApplication
from PySide2.QtUiTools import QUiLoader
from os import path
import rospy
import detector

file_path = path.dirname(__file__)
file_path = file_path.replace('/', '//')

class info_updater():   
    def __init__(self):
        rospy.Subscriber("tello_state", String, self.update_state)
        rospy.Subscriber("tello_image", Image, self.update_img)
        self.con_thread = threading.Thread(target = rospy.spin)
        self.con_thread.start()

    def update_state(self,data):
        global tello_state, tello_state_lock, t_wu, R_wu
        tello_state_lock.acquire() #thread locker
        tello_state = data.data
        statestr = tello_state.split(';')
        #print(statestr)
        for item in statestr:
            if 'mid:' in item:
                pass
            elif 'x:' in item:
                x = int(item.split(':')[-1])
                t_wu[0] = x
            elif 'z:' in item:
                z = int(item.split(':')[-1])
                t_wu[2] = z
            elif 'mpry:' in item:
                pass
            # y can be recognized as mpry, so put y first
            elif 'y:' in item:
                y = int(item.split(':')[-1])
                t_wu[1] = y
            elif 'pitch:' in item:
                pitch = int(item.split(':')[-1])
            elif 'roll:' in item:
                roll = int(item.split(':')[-1])
            elif 'yaw:' in item:
                yaw = int(item.split(':')[-1])
        R_wu = (R.from_euler('zyx', [yaw, pitch, roll], degrees=True)).as_quat()
        #print(t_wu)
        #print(R_wu)
        tello_state_lock.release()

    def update_img(self,data):
        global img, img_lock
        img_lock.acquire()#thread locker
        img = CvBridge().imgmsg_to_cv2(data, desired_encoding = "passthrough")
        img_lock.release()

class main_exe:
    global img

    def __init__(self):
        self.commandPub_ = rospy.Publisher('tello_control', String, queue_size=1)  # 发布tello格式控制信号

        # PyCharm路径
        self.ui = QUiLoader().load(file_path + '//tello.ui')
        # cmd以及打包路径
        # self.ui = QUiLoader().load('tello.ui')

        self.ui.mon.clicked.connect(self.mon)
        self.ui.takeoff.clicked.connect(self.takeoff)
        self.ui.land.clicked.connect(self.land)
        self.ui.save.clicked.connect(self.save)

        self.ui.forward.clicked.connect()
        self.ui.back.clicked.connect()
        self.ui.left.clicked.connect()
        self.ui.right.clicked.connect()
        self.ui.up.clicked.connect()
        self.ui.down.clicked.connect()

        self.ui.command_1.clicked.connect()
        self.ui.command_2.clicked.connect()
        self.ui.command_3.clicked.connect()
        self.ui.command_4.clicked.connect()
        self.ui.command_5.clicked.connect()
        self.ui.command_6.clicked.connect()
        
    def forward(self):
        cm = self.ui.forward_cm.text()
        command = "forward "+(str(cm))
        self.publishCommand(command)
    
    def back(self):
        cm = self.ui.forward_cm.text()
        command = "back "+(str(cm))
        self.publishCommand(command)
    
    def up(self):
        cm = self.ui.up_cm.text()
        command = "up "+(str(cm))
        self.publishCommand(command)
    
    def down(self):
        cm = self.ui.down_cm.text()
        command = "down "+(str(cm))
        self.publishCommand(command)
    
    def right(self):
        cm = self.ui.right_cm.text()
        command = "right "+(str(cm))
        self.publishCommand(command)
    
    def left(self):
        cm = self.ui.left_cm.text()
        command = "left "+(str(cm))
        self.publishCommand(command)

    def cw(self):
        cm = self.ui.cw_degree.text()
        command = "cw "+(str(cm))
        self.publishCommand(command)

    def ccw(self):
        cm = self.ui.ccw_degree.text()
        command = "ccw "+(str(cm))
        self.publishCommand(command)

    def takeoff(self):
        command = "takeoff"
        self.publishCommand(command)
        print ("ready")
        
    def mon(self):
        command = "mon"
        self.publishCommand(command)
        print ("mon")

    def land(self):
        command = "land"
        self.publishCommand(command)

    def command_1(self):
        command = self.ui.command_str_1.text()
        self.publishCommand(command)
        
    def command_2(self):
        command = self.ui.command_str_2.text()
        self.publishCommand(command)
        
    def command_3(self):
        command = self.ui.command_str_3.text()
        self.publishCommand(command)
        
    def command_4(self):
        command = self.ui.command_str_4.text()
        self.publishCommand(command)
        
    def command_5(self):
        command = self.ui.command_str_5.text()
        self.publishCommand(command)
        
    def command_6(self):
        command = self.ui.command_str_6.text()
        self.publishCommand(command)

    def save(self):
        if img is not None:
			img_copy = img.copy()
			rect = img.copy()
			rect = rect[0:40, 0:rect.shape[1]]
			cv2.rectangle(rect, (0, 0), (img_copy.shape[1], 40), (255, 255, 255), -1)
			cv2.putText(rect, tello_state, (0, 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
			img_copy = np.vstack([rect, img_copy])
            detector.recordpic(file_path, 'original', img_copy)
        
    # 向相关topic发布tello命令
    def publishCommand(self, command_str):
        msg = String()
        msg.data = command_str
        self.commandPub_.publish(msg)
        rate = rospy.Rate(0.3)
        rate.sleep()

if __name__ == "__main__":
    rospy.init_node('tello_control', anonymous=True)
    infouper = info_updater()
    # app = QApplication(sys.argv)
    app = QApplication([])

    EXE = main_exe()
    EXE.ui.show()
    # sys.exit(app.exec_())
    app.exec_()
