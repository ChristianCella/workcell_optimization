import tkinter as tk
from tkinter import colorchooser

def pick_color():
    # Open system color chooser
    color = colorchooser.askcolor(title="Pick a Color")

    if color and color[0]:
        r, g, b = [int(x) for x in color[0]]
        rgba = (r/255, g/255, b/255, 1.0)
        print(f"\nSelected Color:")
        print(f"RGB:  ({r}, {g}, {b})")
        print(f"RGBA: ({rgba[0]:.3f}, {rgba[1]:.3f}, {rgba[2]:.3f}, {rgba[3]:.1f})")
        # Update label and window background
        label.config(text=f"RGB: {r},{g},{b}", bg=color[1], fg="white" if r+g+b < 382 else "black")

# Create simple Tkinter UI
root = tk.Tk()
root.title("RGBA Color Picker")
root.geometry("250x150")

label = tk.Label(root, text="Click below to pick a color", pady=20)
label.pack()

btn = tk.Button(root, text="Pick Color", command=pick_color, width=15)
btn.pack(pady=10)

root.mainloop()
