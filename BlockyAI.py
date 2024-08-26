import tkinter as tk
from tkinter import ttk

class Block:
    def __init__(self, canvas, x, y, text, app, input_field=False, resize_field=False, deletable=True):
        self.canvas = canvas
        self.text = text
        self.app = app
        self.input_field = input_field
        self.resize_field = resize_field
        self.deletable = deletable
        self.connected = False
        self.id = canvas.create_rectangle(x, y, x + 100, y + 50, fill="skyblue", outline="")
        self.text_id = canvas.create_text(x + 50, y + 15, text=text, fill="black", font=("Arial", 10, "bold"))
        self.entry = tk.Entry(canvas, bg="white") if input_field or resize_field else None

        
        if self.entry:
            self.entry.place(x=x + 10, y=y + 25, width=80)
        self.bind_events()

    def bind_events(self):
        self.canvas.tag_bind(self.id, "<B1-Motion>", self.on_drag)
        self.canvas.tag_bind(self.text_id, "<B1-Motion>", self.on_drag)
        if self.deletable:
            self.canvas.tag_bind(self.id, "<ButtonRelease-1>", self.on_release)
            self.canvas.tag_bind(self.text_id, "<ButtonRelease-1>", self.on_release)

    def on_drag(self, event):
        x, y = event.x, event.y
        self.canvas.coords(self.id, x - 50, y - 25, x + 50, y + 25)
        self.canvas.coords(self.text_id, x, y - 10)
        if self.entry:
            self.entry.place(x=x - 40, y=y)
        self.app.update_code_display()

    def on_release(self, event):
        self.connected = self.check_connection()
        self.update_transparency()
        if self.deletable and self.app.check_delete_area(event.x, event.y):
            self.app.delete_block(self)

    def check_connection(self):
        for block in self.canvas.find_withtag("block"):
            if block != self.id:
                x1, y1, x2, y2 = self.canvas.coords(block)
                bx1, by1, bx2, by2 = self.canvas.coords(self.id)
                if abs(bx1 - x1) < 10 and abs(by1 - y2) < 10:
                    self.align_with_block(x1, y2)
                    return True
        return False

    def align_with_block(self, x1, y2):
        self.canvas.coords(self.id, x1, y2, x1 + 100, y2 + 50)
        self.canvas.coords(self.text_id, x1 + 50, y2 + 15)
        if self.entry:
            self.entry.place(x=x1 + 10, y=y2 + 25)

    def update_transparency(self):
        fill_color = "lightblue" if not self.connected and self.text != "Starting Block" else "skyblue"
        self.canvas.itemconfig(self.id, fill=fill_color)

    def get_value(self):
        return self.entry.get() if self.entry else None


    
class BlockApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BlockyAI_v1.1")
        self.root.geometry("1600x700")
        self.style = ttk.Style()
        self.style.configure("TFrame", background="whitesmoke")
        self.style.configure("TButton", font=("Arial", 10, "bold"), foreground="black")
        self.style.configure("TLabel", font=("Arial", 12, "bold"))

        self.dataset_var = tk.StringVar(value="fashion_mnist")
        self.optimizer_var = tk.StringVar(value="Adam")
        self.batch_size_var = tk.IntVar(value=32)
        self.epochs_var = tk.IntVar(value=50)
        self.learning_rate_var = tk.DoubleVar(value=0.001)

        self.blocks = []

        self.setup_ui()

    def setup_ui(self):
        self.create_code_frame()
        self.create_canvas()
        self.create_block_holder()
        self.create_settings_frame()

    def create_code_frame(self):
        code_frame = ttk.Frame(self.root)
        code_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10, expand=True)
        ttk.Label(code_frame, text="Code").pack()
        self.code_text = tk.Text(code_frame, bg="lightyellow", font=("Courier", 9))
        self.code_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    def create_canvas(self):
        self.canvas = tk.Canvas(self.root, bg="whitesmoke")
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas.create_text(400, 20, text="Canvas", font=("Arial", 14, "bold"), fill="black")
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def on_canvas_resize(self, event):
        self.update_delete_area(event.width, event.height)

    def create_block_holder(self):
        block_holder = ttk.Frame(self.root)
        block_holder.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        ttk.Label(block_holder, text="Blocks").pack()
        self.add_button(block_holder, "Starting Block", start=True, deletable=False)
        self.add_button(block_holder, "Dense Layer", input_field=True)
        self.add_button(block_holder, "Conv2D Layer", input_field=True)
        self.add_button(block_holder, "Flatten")
        self.add_button(block_holder, "Activation")
        self.add_button(block_holder, "Resizing Layer", resize_field=True)
        self.add_button(block_holder, "AveragePooling2D Layer")
        self.add_button(block_holder, "MaxPooling2D Layer")

    def create_settings_frame(self):
        settings_frame = ttk.Frame(self.root)
        settings_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        ttk.Label(settings_frame, text="Settings").pack()
        self.create_setting_option(settings_frame, "Dataset:", self.dataset_var, ["fashion_mnist", "mnist", "cifar10"])
        self.create_setting_option(settings_frame, "Optimizer:", self.optimizer_var, ["Adam", "SGD"])
        self.create_setting_entry(settings_frame, "Batch Size:", self.batch_size_var)
        self.create_setting_entry(settings_frame, "Epochs:", self.epochs_var)
        self.create_setting_entry(settings_frame, "Learning Rate:", self.learning_rate_var)

    def create_setting_option(self, frame, label, var, options):
        opt_frame = ttk.Frame(frame)
        opt_frame.pack(pady=5)
        ttk.Label(opt_frame, text=label).pack(side=tk.LEFT)
        option_menu = ttk.Combobox(opt_frame, textvariable=var, values=options, state="readonly")
        option_menu.pack(side=tk.LEFT)
        var.trace_add("write", lambda *args: self.update_code_display())

    def check_delete_area(self, x, y):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        return x > canvas_width - 110 and y > canvas_height - 60
    
    def create_setting_entry(self, frame, label, var):
        entry_frame = ttk.Frame(frame)
        entry_frame.pack(pady=5)
        ttk.Label(entry_frame, text=label).pack(side=tk.LEFT)
        ttk.Entry(entry_frame, textvariable=var, width=5).pack(side=tk.LEFT)
        var.trace_add("write", lambda *args: self.update_code_display())

    def update_delete_area(self, canvas_width, canvas_height):
        self.canvas.delete("delete_area")
        self.canvas.create_rectangle(canvas_width - 110, canvas_height - 60, canvas_width -10 , canvas_height -10 , fill="red", tags="delete_area")
        self.canvas.create_text(canvas_width - 60, canvas_height - 35, text="Delete", fill="white", font=("Arial", 10, "bold"), tags="delete_area")

    def add_button(self, parent, text, start=False, input_field=False, resize_field=False, deletable=True):
        button = ttk.Button(parent, text=text, command=lambda: self.add_block(text, input_field, resize_field, deletable))
        button.pack(side=tk.TOP, padx=5, pady=5)
        if start:
            button.state(['disabled'])
            self.add_block(text, input_field, resize_field, deletable)

    def add_block(self, text, input_field, resize_field, deletable):
        block = Block(self.canvas, 50, 50 + len(self.blocks) * 60, text, self, input_field, resize_field, deletable)
        self.blocks.append(block)
        self.canvas.addtag_withtag("block", block.id)
        self.update_code_display()

    def update_code_display(self, *args):
        code = self.generate_code()
        self.code_text.delete("1.0", tk.END)
        self.code_text.insert(tk.END, code)

    def generate_code(self):
        dataset_name = self.dataset_var.get()
        input_shape = "(28, 28, 1)"
        labels = []

        if dataset_name == "mnist":
            input_shape = "(28, 28, 1)"
            labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        elif dataset_name == "cifar10":
            input_shape = "(32, 32, 3)"
            labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        elif dataset_name == "fashion_mnist":
            input_shape = "(28, 28, 1)"
            labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
                      "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

        code = (
            "# Auto-generated Python code\n"
            "# Run this.py in terminal\n"
            "# TensorFlow 2.17.0\n"
            "# Python 3.10.4\n\n"
            "import tensorflow as tf\n"
            "from tensorflow.keras.layers import Dense, Conv2D, Activation, Resizing, AveragePooling2D, MaxPooling2D, Flatten\n"
            "from tensorflow.keras.optimizers import Adam, SGD\n"
            "from tensorflow.keras.callbacks import Callback, EarlyStopping\n"
            "import numpy as np\n"
            "import cv2\n\n"
            "tf.keras.backend.clear_session()\n"
            f"(x_train, y_train), (x_test, y_test) = tf.keras.datasets.{dataset_name}.load_data()\n\n"
            "x_train, x_test = x_train / 255.0, x_test / 255.0\n"
            "if len(x_train.shape) == 3:\n"
            "    x_train = x_train[..., tf.newaxis]\n"
            "    x_test = x_test[..., tf.newaxis]\n\n"
            "num_classes = 10\n"
            "train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes)\n"
            "test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes)\n\n"
            "def build_model(num_classes):\n"
            f"    model = tf.keras.Sequential([\n"
            f"        tf.keras.Input(shape={input_shape}),\n"
        )
        start_connected = False

        for block in sorted(self.blocks, key=lambda b: self.canvas.coords(b.id)[1]):
            text = self.canvas.itemcget(block.text_id, "text")
            if text == "Starting Block":
                start_connected = True

            if start_connected and block.check_connection():
                if text == "Dense Layer":
                    units = block.get_value() or 128
                    code += f"        Dense({units}, activation='relu'),\n"
                elif text == "Conv2D Layer":
                    filters = block.get_value() or 32
                    code += f"        Conv2D({filters}, kernel_size=(3, 3), activation='relu'),\n"
                elif text == "Activation":
                    code += "        Activation('softmax'),\n"
                elif text == "Resizing Layer":
                    size = block.get_value()
                    width, height = map(str.strip, size.split(',')) if size else (64, 64)
                    code += f"        Resizing({width}, {height}),\n"
                elif text == "AveragePooling2D Layer":
                    code += "        AveragePooling2D(pool_size=(2, 2)),\n"
                elif text == "MaxPooling2D Layer":
                    code += "        MaxPooling2D(pool_size=(2, 2)),\n"
                elif text == "Flatten":
                    code += "        Flatten(),\n"

        code += (
            "        Dense(num_classes, activation='softmax')\n"
            "    ])\n"
            "    return model\n\n"
            "model = build_model(num_classes)\n"
        )
        optimizer = self.optimizer_var.get()
        learning_rate = self.learning_rate_var.get()
        batch_size = self.batch_size_var.get()
        epochs = self.epochs_var.get()
        code += (
            f'model.compile(optimizer={optimizer}(learning_rate={learning_rate}), loss="categorical_crossentropy", metrics=["accuracy"])\n'
            "callbacks = [EarlyStopping(monitor='val_loss', patience=3)]\n"
            f"model.fit(x_train, train_one_hot, batch_size={batch_size}, epochs={epochs}, callbacks=callbacks, validation_data=(x_test, test_one_hot))\n"
        )

        code += (
            "prediction = model.predict(x_test)\n\n"
            "N = np.random.randint(0, high=len(x_test), dtype=int)\n"
            f"labels = {labels}\n"
            "print(f'sum: {np.sum(prediction, axis=1)}')\n"
            "print(f'predict index: {np.argmax(prediction, axis=1)}')\n"
            "print(f'Predict: {labels[np.argmax(prediction, axis=1)[N]]}')\n"
            "print(f'Correct: {labels[y_test[N][0]]}')\n\n"
            "image = x_test[N]\n"
            "if image.shape[-1] == 1:\n"
            "    image = image.reshape(image.shape[0], image.shape[1]) * 255\n\n"
            
            "cv2.namedWindow('img',cv2.WINDOW_NORMAL)\n"
            "cv2.imshow('img', image)\n"
            "cv2.resizeWindow('img',300,300)\n"
            "cv2.waitKey(0)\n"
            "cv2.destroyAllWindows()\n"
        )
        
        return code

    def delete_block(self, block):
        self.canvas.delete(block.id)
        self.canvas.delete(block.text_id)
        if block.entry:
            block.entry.destroy()
        self.blocks.remove(block)
        self.update_code_display()

root = tk.Tk()
app = BlockApp(root)
root.mainloop()