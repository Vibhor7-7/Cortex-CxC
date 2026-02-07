import torch
from transformers import pipeline

pipeline = pipeline(
    task="summarization",
    model="google/pegasus-xsum",
    dtype=torch.float16,
    device=0
)
pipeline("""Climate change is one of the most pressing global 
         challenges of the twenty-first century, affecting 
         ecosystems, economies, and human health across the world. 
         Rising global temperatures have led to more frequent and 
         severe weather events such as heatwaves, floods, droughts, 
         and hurricanes, placing strain on infrastructure and food 
         systems. Human activities, particularly the burning of fossil 
         fuels and large-scale deforestation, are the primary drivers of 
         greenhouse gas emissions that accelerate this process. While international 
         agreements and national policies aim to reduce emissions and promote renewable energy, 
         progress remains uneven. Addressing climate change will require coordinated global action, technological innovation, 
         and long-term changes in consumption and production patterns.""")