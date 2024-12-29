import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import requests
import pandas as pd
import json
import os
import logging
from pydantic import create_model, ValidationError
import subprocess
import threading

# Global Configuration
CONFIG = {
    "ollama_endpoint": "http://localhost:11434/api/chat",
    "output_dir": "output",
    "schemas_dir": "schemas",
    "log_file": "dataset_generator.log",
}

# Initialize logging
logging.basicConfig(
    filename=CONFIG["log_file"],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Ensure required folders exist
os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(CONFIG["schemas_dir"], exist_ok=True)

# Global Variables
input_files = []
fields = {"prompt": "Generate data for {example}", "output": ""}
system_prompt = ""
selected_model = ""
configurations = {"batch_size": 10, "row_count": 100}
output_file = os.path.join(CONFIG["output_dir"], "dataset.csv")
schema = {}
df = pd.DataFrame()


# Helper Functions
def log_and_display_error(message):
    logging.error(message)
    messagebox.showerror("Error", message)

def fetch_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode != 0:
            log_and_display_error(f"Failed to run 'ollama list': {result.stderr}")
            return []

        models = [line.split()[0] for line in result.stdout.strip().split("\n")[1:]]
        return models
    except Exception as e:
        log_and_display_error(f"Failed to fetch models: {e}")
        return []

def select_files():
    global input_files
    input_files = filedialog.askopenfilenames(
        title="Select Input Files",
        filetypes=[("Text Files", "*.txt"), ("JSON Files", "*.json"), ("All Files", "*.*")],
    )
    file_list.delete(0, tk.END)
    for file in input_files:
        file_list.insert(tk.END, file)

def set_output_file():
    global output_file
    output_file = filedialog.asksaveasfilename(
        title="Save Output CSV", defaultextension=".csv", filetypes=[("CSV Files", "*.csv")]
    )
    output_entry.delete(0, tk.END)
    output_entry.insert(0, output_file)

def update_schema():
    global schema
    schema_text = schema_entry.get("1.0", tk.END).strip()
    try:
        schema = json.loads(schema_text)
        messagebox.showinfo("Success", "Schema updated successfully!")
        logging.info("Schema updated: %s", schema)
    except json.JSONDecodeError:
        log_and_display_error("Invalid JSON schema!")

def create_pydantic_model(schema):
    try:
        return create_model(
            "DynamicSchema",
            **{
                key: (eval(value.get("type", "str")), ...)
                for key, value in schema.get("properties", {}).items()
            },
        )
    except Exception as e:
        log_and_display_error(f"Error creating Pydantic model: {e}")
        return None

def load_input_files():
    examples = []
    for file in input_files:
        _, ext = os.path.splitext(file)
        try:
            with open(file, "r", encoding="utf-8") as f:
                if ext == ".txt":
                    examples.extend(f.read().splitlines())
                elif ext == ".json":
                    data = json.load(f)
                    examples.extend(data if isinstance(data, list) else [data])
        except Exception as e:
            log_and_display_error(f"Failed to load {file}: {e}")
    return examples

def call_ollama_api(prompts, pydantic_model):
    responses = []
    for prompt in prompts:
        payload = {
            "model": selected_model,
            "messages": [{"role": "user", "content": prompt}],
            "system": system_prompt,
            "format": schema,
        }
        try:
            response = requests.post(CONFIG["ollama_endpoint"], json=payload)
            response.raise_for_status()
            parsed_response = pydantic_model.parse_raw(response.text)
            responses.append(parsed_response.dict())
        except (requests.RequestException, ValidationError) as e:
            logging.error(f"Error with API call or response validation: {e}")
            responses.append({})
    return responses

def generate_dataset():
    global df
    if not input_files or not selected_model or not output_file:
        log_and_display_error("All required fields must be filled!")
        return

    examples = load_input_files()
    if not examples:
        log_and_display_error("No examples found in input files.")
        return

    pydantic_model = create_pydantic_model(schema)
    if not pydantic_model:
        return

    rows_generated = 0
    progress_bar["value"] = 0
    progress_bar["maximum"] = configurations["row_count"]

    while rows_generated < configurations["row_count"]:
        batch = examples[rows_generated : rows_generated + configurations["batch_size"]]
        prompts = [fields["prompt"].format(example=ex) for ex in batch]
        responses = call_ollama_api(prompts, pydantic_model)

        for ex, res in zip(batch, responses):
            data_row = {"input": ex, **res}
            df = pd.concat([df, pd.DataFrame([data_row])], ignore_index=True)
            rows_generated += 1
            progress_bar["value"] = rows_generated

        preview_text.delete("1.0", tk.END)
        preview_text.insert(tk.END, df.head(10).to_string())

    try:
        df.to_csv(output_file, index=False)
        messagebox.showinfo("Success", f"Dataset saved to {output_file}")
        logging.info("Dataset saved: %s", output_file)
    except Exception as e:
        log_and_display_error(f"Failed to save CSV: {e}")


def open_settings_window():
    settings_window = tk.Toplevel(root)
    settings_window.title("Settings")

    tk.Label(settings_window, text="Batch Size:").grid(row=0, column=0, padx=10, pady=5)
    batch_entry = tk.Entry(settings_window)
    batch_entry.insert(0, configurations["batch_size"])
    batch_entry.grid(row=0, column=1, padx=10, pady=5)

    tk.Label(settings_window, text="Row Count:").grid(row=1, column=0, padx=10, pady=5)
    row_entry = tk.Entry(settings_window)
    row_entry.insert(0, configurations["row_count"])
    row_entry.grid(row=1, column=1, padx=10, pady=5)

    def save_settings():
        try:
            configurations["batch_size"] = int(batch_entry.get())
            configurations["row_count"] = int(row_entry.get())
            messagebox.showinfo("Settings", "Settings saved successfully!")
            settings_window.destroy()
        except ValueError:
            log_and_display_error("Invalid input! Batch size and row count must be integers.")

    tk.Button(settings_window, text="Save", command=save_settings).grid(row=2, column=0, columnspan=2, pady=10)


# GUI Layout
root = tk.Tk()
root.title("Advanced Dataset Generator")

# Input File Selection
tk.Label(root, text="Select Input Files:").grid(row=0, column=0, padx=10, pady=5)
file_list = tk.Listbox(root, selectmode=tk.MULTIPLE, width=50)
file_list.grid(row=1, column=0, padx=10, pady=5)
tk.Button(root, text="Browse...", command=select_files).grid(row=1, column=1, padx=10, pady=5)

# Schema Editor
tk.Label(root, text="Schema (JSON):").grid(row=2, column=0, padx=10, pady=5)
schema_entry = tk.Text(root, height=10, width=50)
schema_entry.grid(row=3, column=0, padx=10, pady=5)
tk.Button(root, text="Update Schema", command=update_schema).grid(row=3, column=1, padx=10, pady=5)

# Model Selection
tk.Label(root, text="Select Model:").grid(row=4, column=0, padx=10, pady=5)
models = fetch_models()
model_combobox = ttk.Combobox(root, values=models, state="readonly")
model_combobox.grid(row=4, column=1, padx=10, pady=5)

def set_model():
    global selected_model
    selected_model = model_combobox.get()
    logging.info(f"Selected model: {selected_model}")

model_combobox.bind("<<ComboboxSelected>>", lambda _: set_model())

# Output File Selection
tk.Label(root, text="Output File:").grid(row=5, column=0, padx=10, pady=5)
output_entry = tk.Entry(root, width=50)
output_entry.grid(row=6, column=0, padx=10, pady=5)
tk.Button(root, text="Browse...", command=set_output_file).grid(row=6, column=1, padx=10, pady=5)

# Generate Button and Progress Bar
tk.Button(root, text="Generate Dataset", command=lambda: threading.Thread(target=generate_dataset).start()).grid(row=7, column=0, columnspan=2, pady=10)
progress_bar = ttk.Progressbar(root, length=300, mode="determinate")
progress_bar.grid(row=8, column=0, columnspan=2, pady=5)

# Settings Menu
tk.Button(root, text="Settings", command=open_settings_window).grid(row=9, column=0, columnspan=2, pady=10)

# Live Preview
tk.Label(root, text="CSV Preview:").grid(row=10, column=0, padx=10, pady=5)
preview_text = tk.Text(root, height=10, width=50)
preview_text.grid(row=11, column=0, columnspan=2, padx=10, pady=5)

root.mainloop()
