import tkinter as tk

class Block:
    def __init__(self, canvas, x, y, text, app, input_field=False, resize_field=False, deletable=True):
        self.canvas = canvas
        self.text = text
        self.app = app
        self.input_field = input_field
        self.resize_field = resize_field
        self.deletable = deletable
        self.connected = False
        self.id = canvas.create_rectangle(x, y, x + 100, y + 50, fill="skyblue")
        self.text_id = canvas.create_text(x + 50, y + 15, text=text, fill="black")
        self.entry = tk.Entry(canvas) if input_field or resize_field else None
        if self.entry:
            self.entry.place(x=x + 10, y=y + 30, width=80)
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
        if self.deletable and self.check_delete_area(event.x, event.y):
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
        self.canvas.coords(self.text_id, x1 + 50, y2 + 25)
        if self.entry:
            self.entry.place(x=x1 + 10, y=y2 + 30)

    def update_transparency(self):
        fill_color = "lightblue" if not self.connected and self.text != "Starting Block" else "skyblue"
        self.canvas.itemconfig(self.id, fill=fill_color)

    def get_value(self):
        return self.entry.get() if self.entry else None

    def check_delete_area(self, x, y):
        return x > 700 and y > 550

class BlockApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Block Code Editor")
        self.optimizer_var = tk.StringVar(value="Adam")
        self.batch_size_var = tk.IntVar(value=32)
        self.epochs_var = tk.IntVar(value=10)

        self.blocks = []

        self.setup_ui()
        self.create_delete_area()

    def setup_ui(self):
        self.create_code_frame()
        self.create_canvas()
        self.create_block_holder()
        self.create_settings_frame()

    def create_code_frame(self):
        code_frame = tk.Frame(self.root)
        code_frame.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(code_frame, text="Code", font=("Arial", 14, "bold")).pack()
        self.code_text = tk.Text(code_frame, width=90, height=30, bg="lightyellow", font=("Courier", 9))
        self.code_text.pack(side=tk.LEFT, fill=tk.Y)

    def create_canvas(self):
        self.canvas = tk.Canvas(self.root, width=800, height=600, bg="lightgrey")
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        tk.Label(self.canvas, text="Canvas", font=("Arial", 14, "bold"), bg="lightgrey").place(x=350, y=10)

    def create_block_holder(self):
        block_holder = tk.Frame(self.root)
        block_holder.pack(side=tk.TOP)
        tk.Label(block_holder, text="Blocks", font=("Arial", 14, "bold")).pack()
        self.add_button("Starting Block", start=True, deletable=False)
        self.add_button("Dense Layer", input_field=True)
        self.add_button("Conv2D Layer", input_field=True)
        self.add_button("Flatten")
        self.add_button("Activation")
        self.add_button("Resizing Layer", resize_field=True)
        self.add_button("AveragePooling2D Layer")
        self.add_button("MaxPooling2D Layer")

    def create_settings_frame(self):
        settings_frame = tk.Frame(self.root)
        settings_frame.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Label(settings_frame, text="Settings", font=("Arial", 14, "bold")).pack()
        self.create_setting_option(settings_frame, "Optimizer:", self.optimizer_var, ["Adam", "SGD"])
        self.create_setting_entry(settings_frame, "Batch Size:", self.batch_size_var)
        self.create_setting_entry(settings_frame, "Epochs:", self.epochs_var)

    def create_setting_option(self, frame, label, var, options):
        opt_frame = tk.Frame(frame)
        opt_frame.pack()
        tk.Label(opt_frame, text=label).pack(side=tk.TOP)
        tk.OptionMenu(opt_frame, var, *options, command=self.update_code_display).pack(side=tk.TOP)

    def create_setting_entry(self, frame, label, var):
        entry_frame = tk.Frame(frame)
        entry_frame.pack()
        tk.Label(entry_frame, text=label).pack(side=tk.TOP)
        tk.Entry(entry_frame, textvariable=var, width=5).pack(side=tk.TOP)
        var.trace_add("write", lambda *args: self.update_code_display())

    def create_delete_area(self):
        self.canvas.create_rectangle(700, 550, 800, 600, fill="red")
        self.canvas.create_text(750, 575, text="Delete", fill="white")

    def add_button(self, text, start=False, input_field=False, resize_field=False, deletable=True):
        button = tk.Button(self.root, text=text, command=lambda: self.add_block(text, input_field, resize_field, deletable))
        button.pack(side=tk.TOP)
        if start:
            button.config(state=tk.DISABLED)
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
        code = (
            "# Auto-generated Python code\n"
            "# Run this.py in terminal\n"
            "# TensorFlow 2.17.0\n"
            "# Python 3.10.4\n\n"
            "import tensorflow as tf\n"
            "from tensorflow.keras.layers import Dense, Conv2D, Activation, Resizing, AveragePooling2D, MaxPooling2D, Flatten\n"
            "from tensorflow.keras.optimizers import Adam, SGD\n"
            "from tensorflow.keras.callbacks import Callback, EarlyStopping\n\n"
            "tf.keras.backend.clear_session()\n"
            "(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()\n\n"
            "x_train, x_test = x_train / 255.0, x_test / 255.0\n"
            "x_train = x_train[..., tf.newaxis]\n"
            "x_test = x_test[..., tf.newaxis]\n\n"
            "num_classes = 10\n"
            "train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes)\n"
            "test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes)\n\n"
            "class PrintMetrics(Callback):\n"
            "    def on_epoch_end(self, epoch, logs=None):\n"
            "        print(f'Epoch {epoch+1}: val_loss = {logs['val_loss']}, val_accuracy = {logs['val_accuracy']}')\n\n"
            "def build_model(num_classes):\n"
            "    model = tf.keras.Sequential([\n"
            "        tf.keras.Input(shape=(28, 28, 1)),\n"
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
            "num_classes = 10  # Set this to the number of labels\n"
            "model = build_model(num_classes)\n"
        )
        optimizer = self.optimizer_var.get()
        batch_size = self.batch_size_var.get()
        epochs = self.epochs_var.get()
        code += (
            f'model.compile(optimizer={optimizer}(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])\n'
            "callbacks = [EarlyStopping(monitor='val_loss', patience=3), PrintMetrics()]\n"
            f"model.fit(x_train, train_one_hot, batch_size={batch_size}, epochs={epochs}, callbacks=callbacks, validation_data=(x_test, test_one_hot))\n"
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