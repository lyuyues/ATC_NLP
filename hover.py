# LOL its a dumb program to keep moving my mouse s.t 
# google colab wont disconnect my runtime due to inactivity
import pyautogui
import time
import random

def move_mouse():
    screen_width, screen_height = pyautogui.size()
    while True:
        x = random.randint(0, screen_width - 1)
        y = random.randint(0, screen_height - 1)
        pyautogui.moveTo(x, y, duration=0.5)  
        time.sleep(5) 

if __name__ == "__main__":
    move_mouse()
