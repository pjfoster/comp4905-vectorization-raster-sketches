# comp4905-vectorization-raster-sketches
BCS Honours Project. The code uses image processing techniques to refine raster sketches and transform them into a series of cubic splines. Sobel gradients are used for pixel clustering, while Kruskal's algorithm and Djikstra's algorithm are used for topology extraction and centerline extraction. Algorithms are based on the work of Noris et al.

Made by Patricia Foster
Last updated: April 2017

Vectorization is an important tool for animation, industrial design, and illustration. Unfortunately, most state-of-the-art solutions require expensive hardware and software, and the results leave much to be desired. This project explores current vectorization techniques using traditional image processing as opposed to convolutional neural networks. 

Using nothing but a smartphone or a cheap camera, users should be able to snap a picture of their rough sketch and instantly transform it into a series of smooth cubic splines. 

The code is written as a series of Python scripts, using Open CV and the NumPy, SciPy, NetworkX, and matplotlib librairies. The final vectorization step is done in MATLAB, though the program automatically generates the required MATLAB script. 

The detailed project report, including an assessment of the code's efficiency and the  is included 
