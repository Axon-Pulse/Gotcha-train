# Visualizer

The Axon Visualizer is a Dash-based application designed to display interactive visualizations of Axon-Pulse projects. It offers users the ability to filter, analyze, and display the project's meta-data, and explore the dataset in an intuitive manner.

## Features

- Interactive Data Table: Explore and interact with the data.
- Dynamic filtering and selection mechanisms for dataset exploration.
- Image Visualization: Visualization of images and figures directly within the web interface.
- Customizable data processing and visualization options.

## Installation

Before you start, ensure you have **Python 3.11** installed on your system.

1. Clone this repository or download the source code.
2. Navigate to the `visualizer` directory in your terminal.
3. Install the required dependencies using pip:

```
pip install -r requirements.txt
```

## Usage

To run the Visualizer, follow these steps:

- Prepare your dataset and ensure it's in the expected format. The Visualizer expects a CSV file for metadata and a directory of images.
- Modify the *`manager.py`* to point to your dataset and images directory.
- Run the *`app.py`* script to start the application:

```
python app.py '<config file name>'
```

For example:

```
python app.py 'isar.yaml'
```

- Open a web browser and go to *`http://127.0.0.1:8050/`* to interact with the visualizer.

## Customization

- **`app.py`**: The main running file. contains the basic display (logo, title), and the imports of the visualizer files.
- **`main_layout.py`** and **`main_callbacks.py`**: Modify those files to adjust the layout, style, and functionality of the web interface.
- **`manager.py`**: the classes that proccess and costumize the data to fit the visualizer apps.
- **`callbacks/`** and **`layouts/`**: Directories contains the special layou and callback for each feature of the visualizer.
- **`env_vars.env`**: The file contains the environment variables of the project. can be adjust for the project's needs.

## Interacting with the Visualizer

- 'Data visualization' menu:
  1. Display special filter tool for Dynamic Criteria Selection, Choose the filtered data as a new dataset for further analysis.
  2. Display Bar plot, Histogram and data graph based on selected columns.
- 'Data Analysis' menu: Contains the list of the project's available transforms based on the 'yaml' file of the project. and implement them to the selected images from the data table.

Feel free to adjust the **README** based on your project's specific requirements or additional features you may have added.
