# ArtCaptioning
### DBDBDeep Final Project, Fall 2023 AAI3201

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

In this project, we meticulously selected and processed data sources to train our models effectively. Our

focus was on datasets that provided a rich variety of images along with descriptive captions, which were crucial for teaching our models to interpret and describe visual art. Here’s how we approached this aspect of the project:

### Datasets Used
- **Flickr8k**: Initially, we considered the Flickr8k dataset, known for its diverse collection of 8,000 images, each accompanied by five different captions. This dataset is often used in image captioning tasks due to its rich descriptive content.
- **COCO Dataset**: We ultimately chose the COCO (Common Objects in Context) dataset for its extensive library of over 330,000 images and 200,000 labeled images. COCO is renowned for its variety in image content and the contextual richness of its captions, making it an ideal choice for training our models.

### Data Processing
- **Image Selection and Preprocessing**: From the COCO dataset, we selectively used images that closely aligned with our project's focus on art. This involved filtering for images with artistic elements and ensuring a wide representation of styles and subjects. We then preprocessed these images for optimal compatibility with our AI models, which included resizing, normalization, and augmentation techniques to enhance model training.
- **Caption Processing**: The captions accompanying these images were also processed to ensure consistency and relevance. This involved cleaning and standardizing the text, removing irrelevant or nonsensical captions, and sometimes augmenting the data with additional descriptive language to better suit the artistic context.

### Fine-Tuning for Art Images
- **Custom Data Augmentation**: To better adapt the COCO dataset for our specific use case of interpreting art, we implemented custom data augmentation techniques. This involved artificially creating variations of the art images (like changes in lighting, cropping, and adding minor distortions) to make our model more robust and capable of handling diverse artistic expressions.
- **Specialized Caption Generation**: We also fine-tuned our language generation models to better reflect the language and style commonly used in art descriptions. This included training on art-specific datasets and incorporating art criticism and historical context where relevant, to generate more insightful and accurate captions.

## Model Evaluation

Evaluating the performance of our AI models was a critical component of the ArtCaptioning project. Our evaluation process was designed to assess both the accuracy of image interpretation and the relevance and coherence of the generated captions. Here’s how we conducted the model evaluation:

### Evaluation Metrics
- **Image Interpretation Accuracy**: We used standard image recognition metrics such as Precision, Recall, and F1-Score to evaluate the accuracy of the image interpretation component of our models. These metrics helped us assess how effectively the model identified and understood the key elements and themes in the artworks.
- **Caption Quality Assessment**: For the language generation aspect, we employed BLEU (Bilingual Evaluation Understudy) scores to measure the linguistic quality of the generated captions. BLEU scores are widely used in machine translation and text generation to evaluate the similarity between machine-generated text and reference (human-written) text.

### Comparative Analysis
- **Model-to-Model Comparison**: We conducted comparative analyses between different iterations of our models (ResNet-LSTM, ViT-LSTM, and ViT-GPT-2) to understand the improvements and changes in performance. This involved side-by-side comparisons of the accuracy and quality of the outputs generated by each model configuration.
- **Baseline Comparison**: Additionally, we compared our models' performance against established baselines in the field of image captioning. This helped us contextualize our results within the broader landscape of AI-driven art interpretation.

### Continuous Monitoring and Iteration
- **Feedback Loop**: Continuous monitoring of the model's performance was established, integrating user and expert feedback into ongoing iterations of the model. This iterative process was key to enhancing the model's accuracy and the quality of its output over time.
- **Error Analysis**: We systematically analyzed instances where the model underperformed or generated inaccurate captions. This error analysis was crucial in identifying specific areas for improvement and in refining the model for better performance.

## References and Acknowledgments

### Datasets
- **Flickr8k Dataset**: Details about this dataset can be found at [Flickr8k Dataset Resource](https://example-link-to-flickr8k-dataset.com).
- **COCO Dataset**: More information on the COCO dataset is available at [COCO Dataset Official Website](https://cocodataset.org/).

### Model and Technology References
- **ResNet**: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. [Link to Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).
- **LSTM**: Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735-1780. [Link to Paper](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735).
- **Vision Transformer (ViT)**: Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. [Link to Paper](https://arxiv.org/abs/2010.11929).
- **GPT-2**: Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners. [Link to Paper](https://openai.com/blog/better-language-models/).

### Acknowledgments
- **Project Team**: Special thanks to the entire project team for their relentless efforts and innovative contributions.
