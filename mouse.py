from pynput.mouse import Button, Controller

mouse = Controller()

def moveMouse(coordinate):
    mouse.position = coordinate
