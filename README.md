# unity-geepeetee
This is an attempt to get GPT2 working in Unity by plugging together:
- ONNX Runtime (https://github.com/microsoft/onnxruntime)
- BlingFire tokenizer (https://github.com/microsoft/BlingFire)

## Instructions
- Download the ONNX LM Head model from https://github.com/onnx/models/blob/master/text/machine_comprehension/gpt-2/model/gpt2-lm-head-10.onnx and extract that somewhere OUT OF the unity project directory.
- Get gpt2.i2w and gpt2.bin from Assets\Packages\BlingFireNuget.0.1.8\contentFiles\cs\any and copy those OUT OF the unity project directory.
- Load the Unity project. Look in the hierarchy for the object "NeuralNet", in the inspector fix the paths to where you extracted the aforementioned files.
- Try to hit Play.