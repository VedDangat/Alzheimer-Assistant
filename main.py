from tkinter import *
from tkinter import messagebox
import os
import cv2   # To close OpenCV windows
from face_detection import run_face_detection  # Import function from face_detection.py

NOTES_FILE = "notes.txt"
notes_win = None
text_area = None


# ----------------------------
# Open Notes Window
# ----------------------------
def open_notes_window():
    global notes_win, text_area
    if notes_win and notes_win.winfo_exists():  # Prevent multiple windows
        notes_win.lift()
        return

    notes_win = Toplevel()
    notes_win.title("My Notes / Reminders")
    notes_win.geometry("500x400")

    Label(notes_win, text="Write your reminders here:",
          font=("Arial", 14, "bold")).pack(pady=10)

    # Text box for notes
    text_area = Text(notes_win, wrap=WORD, font=("Arial", 12))
    text_area.pack(expand=True, fill="both", padx=10, pady=10)

    # Load existing notes if available
    if os.path.exists(NOTES_FILE):
        with open(NOTES_FILE, "r") as f:
            text_area.insert(END, f.read())


# ----------------------------
# Save Notes & Close
# ----------------------------
def save_notes_and_close():
    global notes_win, text_area
    if text_area:
        with open(NOTES_FILE, "w") as f:
            f.write(text_area.get("1.0", END))
        messagebox.showinfo("Saved", "Notes saved successfully!")
    if notes_win:
        notes_win.destroy()
        notes_win = None


# ----------------------------
# Exit Application Properly
# ----------------------------
def exit_app(form):
    global notes_win
    try:
        # Close notes window if open
        if notes_win and notes_win.winfo_exists():
            notes_win.destroy()

        # Close OpenCV windows
        cv2.destroyAllWindows()

        # Destroy main window
        form.destroy()

        # Force exit to stop background threads (like camera loop)
        os._exit(0)

    except Exception as e:
        print("Error while exiting:", e)
        os._exit(0)


# ----------------------------
# Main GUI
# ----------------------------
def main():
    form = Tk()
    form.title("Alzheimer's Assistant")
    form.geometry("1200x650")

    # Canvas background
    canvas = Canvas(form, width=1200, height=650, bg="lightblue")
    canvas.pack(fill="both", expand=True)

    # Title label
    header = Label(form, text="Alzheimer's Assistant",
                   font=("Arial", 24, "bold"), bg="lightblue")
    header.place(x=450, y=50)

    # Instructions
    info = Label(form, text="Click the buttons below to start Face Recognition, SOS, RunTime Notes.",
                 font=("Arial", 14), bg="lightblue")
    info.place(x=350, y=120)

    # Start Camera button
    start_btn = Button(form, text="Start Camera",
                       font=("Arial", 14, "bold"),
                       bg="green", fg="white",
                       padx=20, pady=10,
                       command=run_face_detection)
    start_btn.place(x=520, y=220)

    # Notes button
    notes_btn = Button(form, text="My Notes",
                       font=("Arial", 14, "bold"),
                       bg="purple", fg="white",
                       padx=20, pady=10,
                       command=open_notes_window)
    notes_btn.place(x=530, y=300)

    # Save Notes button
    save_btn = Button(form, text="Save Notes",
                      font=("Arial", 14, "bold"),
                      bg="blue", fg="white",
                      padx=20, pady=10,
                      command=save_notes_and_close)
    save_btn.place(x=530, y=360)

    # Exit button (now fully closes everything)
    exit_btn = Button(form, text="Exit",
                      font=("Arial", 14, "bold"),
                      bg="red", fg="white",
                      padx=20, pady=10,
                      command=lambda: exit_app(form))
    exit_btn.place(x=550, y=430)

    form.mainloop()


if __name__ == '__main__':
    main()
