import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
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

# Example Data
example_schema = {
    "type": "object",
    "properties": {
        "role": {"type": "string"},
        "content": {"type": "string"}
    }
}
example_input = [
    {"role": "user", "content": "greg: Collect 10 wood"},
    {"role": "assistant", "content": "Let me see what's nearby... !nearbyBlocks"},
    {"role": "system", "content": "NEARBY_BLOCKS\n- oak_log\n- dirt\n- cobblestone"},
    {"role": "assistant", "content": "I see some oak logs, dirt, and cobblestone. I'll collect oak logs. !collectBlocks(\"oak_log\", 10)"}
]

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

def extract_fields_from_example(example):
    try:
        fields = []
        for item in example:
            if 'content' in item:
                fields.append({'type': 'string', 'name': item['role']})
        return fields
    except Exception as e:
        log_and_display_error(f"Error extracting fields: {e}")
        return []

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

def create_tooltip(widget, text):
    tooltip = tk.Label(root, text=text, background="yellow", wraplength=150)
    def show_tooltip(event):
        tooltip.place(x=event.x_root, y=event.y_root)
    def hide_tooltip(event):
        tooltip.place_forget()
    widget.bind("<Enter>", show_tooltip)
    widget.bind("<Leave>", hide_tooltip)

def open_settings_window():
    settings_window = tk.Toplevel(root)
    settings_window.title("Settings")

    tk.Label(settings_window, text="Batch Size:").grid(row=0, column=0, padx=10, pady=5)
    batch_entry = tk.Entry(settings_window)
    batch_entry.insert(0, configurations["batch_size"])
    batch_entry.grid(row=0, column=1, padx=10, pady=5)
    create_tooltip(batch_entry, "Number of rows to process in each batch.")

    tk.Label(settings_window, text="Row Count:").grid(row=1, column=0, padx=10, pady=5)
    row_entry = tk.Entry(settings_window)
    row_entry.insert(0, configurations["row_count"])
    row_entry.grid(row=1, column=1, padx=10, pady=5)
    create_tooltip(row_entry, "Total number of rows to generate.")

    tk.Label(settings_window, text="Output Directory:").grid(row=2, column=0, padx=10, pady=5)
    output_dir_entry = tk.Entry(settings_window)
    output_dir_entry.insert(0, CONFIG["output_dir"])
    output_dir_entry.grid(row=2, column=1, padx=10, pady=5)
    create_tooltip(output_dir_entry, "Directory where the output files will be saved.")

    def save_settings():
        try:
            configurations["batch_size"] = int(batch_entry.get())
            configurations["row_count"] = int(row_entry.get())
            CONFIG["output_dir"] = output_dir_entry.get()
            messagebox.showinfo("Settings", "Settings saved successfully!")
            settings_window.destroy()
        except ValueError:
            log_and_display_error("Invalid input! Batch size and row count must be integers.")

    tk.Button(settings_window, text="Save", command=save_settings).grid(row=3, column=0, columnspan=2, pady=10)

def open_system_prompt_window():
    prompt_window = tk.Toplevel(root)
    prompt_window.title("Set System Prompt")

    tk.Label(prompt_window, text="System Prompt:").grid(row=0, column=0, padx=10, pady=5)
    prompt_entry = tk.Text(prompt_window, height=10, width=50)
    prompt_entry.grid(row=1, column=0, padx=10, pady=5)
    prompt_entry.insert(tk.END, system_prompt)
    create_tooltip(prompt_entry, "Enter the system prompt for the Ollama model.")

    def save_system_prompt():
        global system_prompt
        system_prompt = prompt_entry.get("1.0", tk.END).strip()
        messagebox.showinfo("System Prompt", "System prompt saved successfully!")
        prompt_window.destroy()

    tk.Button(prompt_window, text="Save", command=save_system_prompt).grid(row=2, column=0, pady=10)

def open_interaction_window():
    interaction_window = tk.Toplevel(root)
    interaction_window.title("Ollama Interaction")

    interaction_text = ScrolledText(interaction_window, height=20, width=80)
    interaction_text.grid(row=0, column=0, padx=10, pady=5)
    create_tooltip(interaction_text, "Visualize the interactions with the Ollama model.")

    def update_interaction_text():
        interaction_text.delete("1.0", tk.END)
        interaction_text.insert(tk.END, "Interaction log will be displayed here...")

    tk.Button(interaction_window, text="Refresh", command=update_interaction_text).grid(row=1, column=0, pady=10)

def generate_schema_from_example():
    example_text = example_entry.get("1.0", tk.END).strip()
    if not example_text:
        log_and_display_error("Example input cannot be empty!")
        return

    try:
        example = json.loads(example_text)
    except json.JSONDecodeError:
        log_and_display_error("Invalid JSON example input!")
        return

    fields = extract_fields_from_example(example)

    payload = {
        "model": selected_model,
        "messages": [{"role": "user", "content": json.dumps(fields)}],
        "system": system_prompt,
    }

    try:
        response = requests.post(CONFIG["ollama_endpoint"], json=payload)
        response.raise_for_status()
        formatted_schema = response.json()  # Assuming the API returns the formatted schema in JSON
        schema_entry.delete("1.0", tk.END)
        schema_entry.insert(tk.END, json.dumps(formatted_schema, indent=4))
        returned_schema_text.delete("1.0", tk.END)
        returned_schema_text.insert(tk.END, json.dumps(formatted_schema, indent=4))
        messagebox.showinfo("Success", "Schema generated successfully!")
        logging.info("Schema generated: %s", formatted_schema)
    except requests.RequestException as e:
        log_and_display_error(f"Error generating schema: {e}")

def append_returned_schema():
    returned_schema = returned_schema_text.get("1.0", tk.END).strip()
    if not returned_schema:
        log_and_display_error("No returned schema to append!")
        return

    try:
        new_schema = json.loads(returned_schema)
    except json.JSONDecodeError:
        log_and_display_error("Invalid JSON in returned schema!")
        return
    
    schema_entry.delete("1.0", tk.END)
    schema_entry.insert(tk.END, json.dumps(new_schema, indent=4))

# GUI Layout
root = tk.Tk()
root.title("Advanced Dataset Generator")
root.geometry("800x600")

# Create main frame with scrollbar
main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=1)

canvas = tk.Canvas(main_frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

second_frame = ttk.Frame(canvas)
canvas.create_window((0,0), window=second_frame, anchor="nw")

# Input File Selection
ttk.Label(second_frame, text="Select Input Files:").grid(row=0, column=0, padx=10, pady=5)
file_list = tk.Listbox(second_frame, selectmode=tk.MULTIPLE, width=50)
file_list.grid(row=1, column=0, padx=10, pady=5)
ttk.Button(second_frame, text="Browse...", command=select_files).grid(row=1, column=1, padx=10, pady=5)

# Schema Editor
ttk.Label(second_frame, text="Schema (JSON):").grid(row=2, column=0, padx=10, pady=5)
schema_entry = ScrolledText(second_frame, height=10, width=50)
schema_entry.insert(tk.END, json.dumps(example_schema, indent=4))
schema_entry.grid(row=3, column=0, padx=10, pady=5)
ttk.Button(second_frame, text="Update Schema", command=update_schema).grid(row=3, column=1, padx=10, pady=5)

# Example Input for Schema Generation
ttk.Label(second_frame, text="Example Input:").grid(row=2, column=2, padx=10, pady=5)
example_entry = ScrolledText(second_frame, height=10, width=50)
example_entry.insert(tk.END, json.dumps(example_input, indent=4))
example_entry.grid(row=3, column=2, padx=10, pady=5)
ttk.Button(second_frame, text="Generate Schema", command=generate_schema_from_example).grid(row=3, column=3, padx=10, pady=5)

# Returned Schema Display and Append Button
ttk.Label(second_frame, text="Returned Schema:").grid(row=4, column=0, padx=10, pady=5)
returned_schema_text = ScrolledText(second_frame, height=10, width=50)
returned_schema_text.grid(row=5, column=0, padx=10, pady=5)
ttk.Button(second_frame, text="Append Returned Schema", command=append_returned_schema).grid(row=5, column=1, padx=10, pady=5)

# Model Selection
ttk.Label(second_frame, text="Select Model:").grid(row=6, column=0, padx=10, pady=5)
models = fetch_models()
model_combobox = ttk.Combobox(second_frame, values=models, state="readonly")
model_combobox.grid(row=7, column=0, padx=10, pady=5)
create_tooltip(model_combobox, "Select the model to use for data generation.")

def set_model():
    global selected_model
    selected_model = model_combobox.get()
    logging.info(f"Selected model: {selected_model}")

model_combobox.bind("<<ComboboxSelected>>", lambda _: set_model())

# Output File Selection
ttk.Label(second_frame, text="Output File:").grid(row=8, column=0, padx=10, pady=5)
output_entry = ttk.Entry(second_frame, width=50)
output_entry.insert(0, output_file)
output_entry.grid(row=9, column=0, padx=10, pady=5)
ttk.Button(second_frame, text="Browse...", command=set_output_file).grid(row=9, column=1, padx=10, pady=5)
create_tooltip(output_entry, "Path where the generated CSV will be saved.")

# Generate Button and Progress Bar
ttk.Button(second_frame, text="Generate Dataset", command=lambda: threading.Thread(target=generate_dataset).start()).grid(row=10, column=0, columnspan=2, pady=10)
progress_bar = ttk.Progressbar(second_frame, length=300, mode="determinate")
progress_bar.grid(row=11, column=0, columnspan=2, pady=5)

# Settings Menu
ttk.Button(second_frame, text="Settings", command=open_settings_window).grid(row=12, column=0, columnspan=2, pady=10)

# System Prompt Window
ttk.Button(second_frame, text="Set System Prompt", command=open_system_prompt_window).grid(row=13, column=0, columnspan=2, pady=10)

# Interaction Window
ttk.Button(second_frame, text="Visualize Interaction", command=open_interaction_window).grid(row=14, column=0, columnspan=2, pady=10)

# Live Preview
ttk.Label(second_frame, text="CSV Preview:").grid(row=15, column=0, padx=10, pady=5)
preview_text = ScrolledText(second_frame, height=10, width=50)
preview_text.grid(row=16, column=0, columnspan=2, padx=10, pady=5)

# Full Request and Response Display
ttk.Label(second_frame, text="Full Request:").grid(row=17, column=0, padx=10, pady=5)
request_text = ScrolledText(second_frame, height=10, width=50)
request_text.grid(row=18, column=0, padx=10, pady=5)
create_tooltip(request_text, "Full content of the request sent to Ollama.")

ttk.Label(second_frame, text="Full Response:").grid(row=17, column=1, padx=10, pady=5)
response_text = ScrolledText(second_frame, height=10, width=50)
response_text.grid(row=18, column=1, padx=10, pady=5)
create_tooltip(response_text, "Full content of the response received from Ollama.")

root.mainloop()
