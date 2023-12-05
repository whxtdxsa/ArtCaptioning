# ArtCaptioning
### DBDBDeep Final Project

## Introduction
ArtCaptioning is an innovative project aimed at translating visual art into vocal interpretations using advanced AI techniques. This project represents a unique intersection of technology and art, exploring new frontiers in AI applications.

## Project Overview
The ArtCaptioning project focuses on using AI models to understand and interpret visual artworks and convert these interpretations into spoken words. This approach opens new possibilities in how we interact with and understand art through technology.

## Model Structure

This project underwent several iterations in its model structure to achieve the optimal balance between accurate image interpretation and effective vocal translation. Below are the key stages of our model development:

### Initial Model: ResNet and LSTM
- **ResNet (Residual Neural Network)**: We initially employed ResNet for its proven effectiveness in image recognition tasks. This deep neural network, known for its ability to train hundreds of layers without performance degradation, was crucial in accurately identifying key elements and nuances within artworks.
- **LSTM (Long Short-Term Memory)**: Paired with ResNet, we used LSTM networks for generating descriptive captions. LSTMs are a type of recurrent neural network (RNN) particularly suited for sequential data. In this context, they were tasked with translating the visual data processed by ResNet into coherent, descriptive language.

### Changed Model: Transformer (ViT) and LSTM
- **Vision Transformer (ViT)**: As the project progressed, we shifted to using Vision Transformers for image recognition. ViT represents a significant advancement in computer vision, bringing the benefits of transformer models (originally designed for NLP tasks) to the realm of image processing. This change was aimed at harnessing the transformer's ability to capture global dependencies in the image data.
- **Integration with LSTM**: We retained the LSTM for caption generation, allowing us to leverage its sequential data processing capabilities while enhancing the image recognition component with ViT.

### Final Model: Transformer (ViT) and Transformer (GPT-2)
- **GPT-2 Integration**: In our final iteration, we introduced another transformer model, GPT-2, for language generation. GPT-2, known for its powerful language understanding and generation capabilities, brought a significant improvement in the quality and relevance of the generated captions.
- **End-to-End Transformer-Based Architecture**: With both ViT and GPT-2, our model became a fully transformer-based system. This architecture allowed for more advanced understanding and processing of both visual and textual data, leading to more accurate and contextually relevant vocal interpretations of art.

## Data Sources and Processing
- **Datasets Used**: Experimented with Flickr8k and COCO datasets for model training and fine-tuning.
- **Data Processing**: Emphasis on processing COCO dataset to align with the project's objectives.

## Experimental Plan
Detailed the experimental approach, including model training, testing, and iteration processes.

## Fine-Tuning for Art Images
Specific strategies employed for fine-tuning the AI models to better interpret and vocalize art images.

## Model Evaluation
An evaluation of the model's performance, including accuracy, reliability, and areas for improvement.

## Results and Insights
A summary of the project's outcomes, highlighting key successes and learnings.

## Future Directions
Potential future enhancements and applications of the project, exploring how it can evolve and expand its impact.

## Contribution
Details on how others can contribute to the project, including guidelines for code contributions, feedback, and collaboration.

## Acknowledgements
Acknowledging individuals, organizations, or resources that played a pivotal role in the project's development.
