# comp4905-vectorization-raster-sketches
BCS Honours Project. The code uses image processing techniques to refine raster sketches and transform them into a series of cubic splines. Sobel gradients are used for pixel clustering, while Kruskal's algorithm and Djikstra's algorithm are used for topology extraction and centreline extraction. Algorithms are based on the work of Noris et al.

Made by Patricia Foster

Last updated: April 2017

--- INTRODUCTION

Vectorization is an important tool for animation, industrial design, and illustration. Unfortunately, most state-of-the-art solutions require expensive hardware and software, and the results leave much to be desired. This project explores current vectorization techniques using traditional image processing as opposed to convolutional neural networks. 

Using nothing but a smartphone or a cheap camera, users should be able to snap a picture of their rough sketch and instantly transform it into a series of smooth cubic splines. 

--- IMPLEMENTATION DETAILS

The code is written as a series of Python 2.7 scripts, using OpenCV and the NumPy, SciPy, NetworkX, and matplotlib librairies. The final vectorization step is done in MATLAB, though the program automatically generates the required MATLAB script. 

The program can be run from any Python IDE or console. The name of the image to be vectorized must be hardcoded into the main.py file. This is definitely not ideal, but it sufficed for the purpose of this project. 

Steps include:

	. Pre-processing with Gaussian filtering to remove noise
	
	. Clustering pixels, using Sobel X and Y gradients to determine the direction of the centreline
	
	. Transforming the clustered pixels into a graph, and pruning it into a MST with Kruskal's algorithm
	
	. Seperating pixels into distinct components based on the location of endpoints and junctions
	
	. Using corners / interest points to make the seperation process more accurate
	
	. Iteratively smoothing individual curves with a Gaussian smoothing operator
	
	. Fitting curves to each seperate point set using MATLAB's curb fitting library

The detailed project report is included in the Documentation folder and contains a detailed explanation of the algorithms used and an assessment of their efficiency and accuracy. 
