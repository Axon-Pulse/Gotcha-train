import gradio as gr
import pandas as pd
from PIL import Image
from torch.nn.functional import softmax


def BasicGradioApp(model, datamodule, trainer, csv_path):
    # Load data
    df = pd.read_csv(csv_path)
    list_of_ids = df["id"].astype(int).tolist()

    # Define the prediction function
    def predict_from_csv(sample_id):
        # Convert string ID back to int if necessary
        sample_id = int(sample_id)

        # Find the corresponding image path
        image_path = df[df["id"] == sample_id]["image_path"].values[0]
        image = Image.open(image_path)

        # Setting up for prediction
        predict_dataloader = datamodule.predict_dataloader()

        # Making a prediction
        predictions = trainer.predict(model, dataloaders=predict_dataloader)
        print(predictions)

        # Assume we return the first prediction's max index (class label)
        index = list_of_ids.index(sample_id)
        print(index)
        predicted_label = predictions[list_of_ids.index(sample_id)]

        # Return both image and prediction
        return image, predicted_label

    # Create Gradio interface components
    sample_dropdown = gr.Dropdown(choices=list_of_ids, label="Select Sample ID")
    output_image = gr.Image(label="Sample Image")
    output_text = gr.Textbox(label="Model dict")
    output_text = gr.Textbox(label="Model Prediction")

    # Create Gradio interface
    iface = gr.Interface(
        fn=predict_from_csv,
        inputs=[sample_dropdown],
        outputs=[output_image, output_text],
        title="Model Prediction Interface",
        description="Select a sample ID to view the image and predict using the model.",
    )
    iface.launch()
