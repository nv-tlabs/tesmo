# Model Overview

## Description:
TeSMo generates a three-dimensional (3D) character animation of a person interacting with objects in a scene based on a text prompt. It contains two main stages (models) to do this, first a Navigation model to generate a root path in the scene, and second an Interaction model to carry out full interactions.

This model is for research and development only.

### License/Terms of Use: 
Non-commercial NSCL License

## Reference:
"Generating Human Interaction Motions in Scenes with Text Control", Yi et al., European Conference on Computer Vision (ECCV), 2024. [[Link]](https://research.nvidia.com/labs/toronto-ai/tesmo/)
<br> 

## Model Architecture: 
**Architecture Type:** Diffusion Model <br>
**Network Architecture:** Transformer Encoder <br>

## Input:
**Input Type(s):** Text, End Goal Location, Scene Geometry <br>
**Input Format(s):** String, Vector, Floor Map and Mesh <br>
**Input Parameters:** 1D, 5D (x, y, z, cos(heading), sin(heading)), 2D image for floormap and vertex/face matrix for meshes <br>
**Other Properties Related to Input:** Text describes the person's action in the scene. End Goal Location is provided separately for both the Navigation and Interaction stages of the model. Scene Geometry is converted to a floormap for the Navigation part while the Interaction uses the full scene mesh.

## Output:
**Output Type(s):**  Root Trajectory, Full-body Joint Positions  <br>
**Output Format:**  Matrix, Matrix <br>
**Output Parameters:** 2D, 3D <br>
**Other Properties Related to Output:** Root Trajectory is the output of the Navigation part of the model (the first stage). Joint Positions are output from the Interaction part (the second stage). Output joint positions are for the SMPL body model. <br> 

## Software Integration:
**Runtime Engine(s):** 
* PyTorch <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
* NVIDIA Ampere <br>
* NVIDIA Hopper <br>
* NVIDIA Lovelace <br>
* NVIDIA Pascal <br>
* NVIDIA Turing <br>
* NVIDIA Volta <br>

**[Preferred/Supported] Operating System(s):** <br>
* Linux

## Model Version(s): 
* tesmo_navigation_v1: model for first stage of the pipeline that generates a root trajectory
* tesmo_interaction_v1 : model for second stage of the pipeline that generates full body motion


## Training and Evaluation Datasets:
* Loco-3D-FRONT (for navigation model) 
* SAMP (for interaction model)

**Data Collection Method by dataset**
* Loco-3D-FRONT and SAMP: Automatic/Sensors

**Labeling Method by dataset**
* Loco-3D-FRONT and SAMP: Hybrid: Automatic/Sensors, Human

**Properties per Datasets:** 
* Loco-3D-FRONT: Created from HumanML3D and 3D-FRONT. Contains 9500 locomotion sequences, each placed in 10 different 3D scenes with a corresponding text description. Roughly 1000 sequences held out for the evaluation split.
* SAMP: Updates the original SAMP datasets with added text description for each motion and additional augmented objects from 3D-FRONT. Motions are extracted from 80 sitting sequences. The training and eval splits follow the original SAMP dataset.

## Inference:
**Engine:** PyTorch <br>
**Test Hardware:** <br>
* RTX 3090

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).