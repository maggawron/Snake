import keyboard

def pressed(event):
    print(event.name)

def main():
    keyboard.on_press(pressed)

    while True:
        key = keyboard.read_key()


# check
if __name__ == "__main__":
    main()