using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using BlingFire;
using Microsoft.ML.OnnxRuntime;
using System;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Linq;

public class NeuralNetTest : MonoBehaviour
{
    public const int VOCAB_SIZE = 50257;

    //Tokeniser model comes from BlingFire https://github.com/microsoft/BlingFire
    public string tokeniserModelPath = "D:/Models/tokenizers/gpt2";
    ulong tokenizerHandle = 0;
    ulong tokenizerI2WHandle = 0;

    //Models come from https://github.com/onnx/models/tree/master/text/machine_comprehension/gpt-2
    //string gpt2ModelPath = "D:/Models/onnx/gpt2-10.onnx";
    public string gpt2ModelPath = "D:/Models/onnx/gpt2-lm-head-10.onnx";
    InferenceSession inferenceSession = null;

    public TMPro.TextMeshProUGUI uiGPT2Status;
    public TMPro.TextMeshProUGUI uiTokenizerStatus;
    public TMPro.TMP_InputField uiInput;
    public TMPro.TextMeshProUGUI uiPredictionOuput;

    #region Model loading & UI

    void Start()
    {
        LoadAll();
    }

    private void Update()
    {
        UpdateUI();
    }

    public void LoadAll()
    {
        LoadTokenizer();
        LoadGPT2();
    }

    public void LoadTokenizer()
    {
        Debug.Log($"Using BlingFire {BlingFireUtils.GetBlingFireTokVersion()}");
        Debug.Log($"Loading the tokenizer from {tokeniserModelPath}");
        tokenizerHandle = BlingFireUtils.LoadModel($"{tokeniserModelPath}.bin");
        tokenizerI2WHandle = BlingFireUtils.LoadModel($"{tokeniserModelPath}.i2w");
        BlingFireUtils.SetNoDummyPrefix(tokenizerHandle, true);
        BlingFireUtils.SetNoDummyPrefix(tokenizerI2WHandle, true);
    }

    public void LoadGPT2()
    {
        Debug.Log($"Loading the gpt2 model from {gpt2ModelPath}");
        inferenceSession = new InferenceSession(gpt2ModelPath);
    }

    public void UpdateUI()
    {
        if (tokenizerHandle != 0 && tokenizerI2WHandle != 0)
        {
            uiTokenizerStatus.text = "Loaded";
            uiTokenizerStatus.color = Color.green;
        } 
        else
        {
            uiTokenizerStatus.text = "Not loaded";
            uiTokenizerStatus.color = Color.red;
        }

        if (inferenceSession != null)
        {
            uiGPT2Status.text = "Loaded";
            uiGPT2Status.color = Color.green;
        }
        else
        {
            uiGPT2Status.text = "Not loaded";
            uiGPT2Status.color = Color.red;
        }
    }

    public void OnGenerateClick()
    {
        string ui_output_str = $"Input: {uiInput.text}\n\n";
        int[] input_tokens = Tokenize(uiInput.text);
        ui_output_str += $"Tokens: [{string.Join(",", input_tokens)}]\n\n";
        var mdl_output = GetOutput1(input_tokens, 5);
        for (int i = 0; i < mdl_output.Count; i++)
        {
            ui_output_str += $"Model output ({i}):\n {mdl_output[i].Aggregate("", (s, p) => s + '\t' + Prediction2Str(p) + "\n")}\n";

        }
        uiPredictionOuput.text = ui_output_str;
    }
    #endregion

    #region Simple Inference Logic
    public class Prediction
    {
        public int token;
        public float confidence;
    }

    public int[] Tokenize(string input_str)
    {
        byte[] inBytes = System.Text.Encoding.UTF8.GetBytes(input_str);
        int[] ids = new int[128];
        int outputCount = BlingFireUtils.TextToIds(tokenizerHandle, inBytes, inBytes.Length, ids, inBytes.Length, 0);
        ids = ids.Take(outputCount).ToArray();
        return ids;
    }

    public List<List<Prediction>> GetOutput1(int[] input_ids, int top_k = -1)
    {
        Tensor<Int64> input_tensor = new DenseTensor<Int64>(new[] { 1, input_ids.Length, 1 });
        for (int i = 0; i < input_ids.Length; i++)
        {
            input_tensor[0, i, 0] = i<input_ids.Length?input_ids[i]:0;
        }

        var model_inputs = new List<NamedOnnxValue>()
        {
            NamedOnnxValue.CreateFromTensor("input1", input_tensor)
        };
        var model_outputs = inferenceSession.Run(model_inputs);

        var token_activation_output = model_outputs.First((v) => v.Name == "output1").AsTensor<float>();
        Debug.Log($"Got an ouput tensor [{String.Join(",", token_activation_output.Dimensions.ToArray())}]");

        List<List<Prediction>> output_predictions = new List<List<Prediction>>();
        for (int i = 0; i < input_ids.Length; i++)
        {
            var logits = token_activation_output.AsEnumerable<float>().Skip(i * VOCAB_SIZE).Take(VOCAB_SIZE);
            float sum = logits.Sum(x => (float)Math.Exp(x));
            IEnumerable<float> softmax = logits.Select(x => (float)Math.Exp(x) / sum);
            var test_sorted_predictions = softmax.Select((x, i) => new Prediction() { token = i, confidence = x }).OrderByDescending(x => x.confidence).Take(top_k>0?top_k: VOCAB_SIZE);
            output_predictions.Add(test_sorted_predictions.ToList());
        }
        return output_predictions;
    }
    public string Prediction2Str(Prediction p)
    {
        var str = BlingFireUtils.IdsToText(tokenizerI2WHandle, new int[] { p.token });
        str = str.Substring(0, str.Length - 1);
        return $"{p.token} ({str}) => {p.confidence}";
    }

    #endregion

    #region Generation Logic (Beam search & co)
    public class PredictionTree
    {
        public Prediction state;
        public PredictionTree parent;
        public List<PredictionTree> next_tokens;


        public PredictionTree()
        {
            state = null;
            parent = null;
            next_tokens = new List<PredictionTree>();
        }

        public PredictionTree AddNextToken(Prediction p)
        {
            PredictionTree pt = new PredictionTree()
            {
                state = p,
                parent = this,
                next_tokens = new List<PredictionTree>()
            };
            next_tokens.Add(pt);
            return pt;
        }
    }
    public List<int> GetPreviousTokens(PredictionTree t)
    {
        var tokens = new List<int>();
        var currentTree = t;
        while(currentTree?.state != null)
        {
            tokens.Add(currentTree.state.token);
            currentTree = currentTree.parent;
        }
        tokens.Reverse();
        return tokens;
    }

    public PredictionTree String2Tree(string str, bool return_root_instead_of_leaf = false)
    {
        var tokens = Tokenize(str);
        var root = new PredictionTree();
        var current_node = root;
        for (int i = 0; i < tokens.Length; i++)
        {
            current_node = current_node.AddNextToken(new Prediction() { token = tokens[i], confidence = 1.0f });
        }
        return return_root_instead_of_leaf?root:current_node;
    }

    public void BuildPredictionsTree(PredictionTree tree, int top_k, int current_depth, int max_depth)
    {
        if (current_depth < max_depth)
        {
            var input_ids = GetPreviousTokens(tree);
            var most_probable_next_tokens = GetOutput1(input_ids.ToArray(), top_k).Last();
            foreach (var next_prediction in most_probable_next_tokens)
            {
                var next_token = tree.AddNextToken(next_prediction);
                BuildPredictionsTree(next_token, top_k, current_depth + 1, max_depth);
            }
        }
    }

    public string PredictWordsGreedy(string input_str, int word_count)
    {
        var input_leaf = String2Tree(input_str);
        BuildPredictionsTree(input_leaf, 1, 0, word_count);
        var current_node = input_leaf.next_tokens.First();
        var tokens = new List<int>();
        while(current_node != null && current_node.next_tokens.Count > 0)
        {
            tokens.Add(current_node.state.token);
            current_node = current_node.next_tokens.First();
        }
        return input_str + BlingFireUtils.IdsToText(tokenizerI2WHandle, tokens.ToArray());
    }

    public void Cleanup()
    {
        BlingFireUtils.FreeModel(tokenizerHandle);
    }
    #endregion
}
