# Large Languaeg Models (LLMs)
Here we learn about:
* Basics of LLMs
* Prompting Techniques
* Training and Decoding
* Dangers of LLMs based Technology Deployment
* Upcoming Cutting Edge Technologies

# Fundamentals
## What is a Large Language Model?
A language model (LM) is a **probabilistic model of text**.

The LLM gives a probability to every word in its vocabularity of appearing next.

"Large" in "large language model" (**L**LM) refers to # of parameters; no agreed-upon treshold

In the term "large language model" (LLM), the word "large" refers to the number of parameters the model has. Let's break down what this means and why there's no agreed-upon threshold for what counts as "large."

### What Are Parameters?

Parameters in a language model are like adjustable settings that help the model make predictions. These settings are tuned during the training process to improve the model's performance on tasks like understanding text, generating sentences, or answering questions. The more parameters a model has, the more complex patterns it can learn from the data.

### Why Is "Large" Important?

1. **Learning Capacity**: More parameters mean the model can learn and store more information, which can improve its ability to understand and generate human-like text.
2. **Performance**: Larger models often perform better on a wide range of tasks because they can capture more nuances and details from the training data.

### No Agreed-Upon Threshold

There isn't a specific number of parameters that universally defines a model as "large." This is because:

1. **Rapid Advancements**: The field of machine learning is advancing quickly. What was considered a large model a few years ago might now be seen as small compared to the latest models.
   
2. **Context Matters**: Different applications and fields might have different standards for what counts as "large." A model that is large for one task might be small for another.

### Real-Life Example

Think of parameters like ingredients in a recipe. If you're making a simple dish, you might only need a few ingredients. But if you're making a complex gourmet meal, you'll need many more ingredients to get the flavors just right. Similarly, a simple model might only need a few parameters, while a more complex model (like those used in advanced language processing) requires many more.

### Example in Numbers

- **Small Model**: A small language model might have around 10 million parameters.
- **Medium Model**: A medium-sized model could have hundreds of millions of parameters.
- **Large Model**: Today's large language models, like GPT-4, can have tens of billions or even hundreds of billions of parameters.

In summary, "large" in the context of large language models refers to the high number of parameters that the model has, enabling it to learn and perform complex tasks. There's no fixed number that defines "large" because the field is always evolving and different contexts might have different requirements.

### Here are some common parameters in language models:

### 1. **Weights**
- **Definition**: These are the core parameters in neural networks. They are the connections between neurons in different layers.
- **Function**: Weights determine how much influence one neuron has on another. During training, the model adjusts these weights to minimize errors in its predictions.

### 2. **Biases**
- **Definition**: Biases are additional parameters in each neuron.
- **Function**: They allow the model to fit the training data better by adding an extra degree of freedom. Biases help the model to shift the activation function to fit the data more accurately.

### 3. **Activation Functions**
- **Definition**: Functions applied to the outputs of neurons before passing them to the next layer.
- **Examples**: Common activation functions include ReLU (Rectified Linear Unit), Sigmoid, and Tanh.
- **Function**: Activation functions introduce non-linearity into the model, allowing it to learn more complex patterns.

### 4. **Learning Rate**
- **Definition**: A hyperparameter (a parameter set before the training process begins) that determines how much the model's parameters are adjusted with respect to the loss gradient.
- **Function**: It controls how quickly the model learns. A high learning rate might make the model learn too quickly and miss the optimal solution, while a low learning rate might make the training process very slow.

### 5. **Dropout Rate**
- **Definition**: A hyperparameter that specifies the fraction of neurons to drop (set to zero) during training.
- **Function**: Dropout helps prevent overfitting (when the model performs well on training data but poorly on new data) by randomly ignoring some neurons during each training step, which encourages the model to learn more robust features.

### 6. **Batch Size**
- **Definition**: The number of training samples used in one iteration of training.
- **Function**: Affects the stability and speed of the training process. Larger batch sizes can lead to faster, but less stable convergence, while smaller batch sizes can lead to more stable, but slower training.

### Example with GPT-3

Let's use GPT-3, one of the most well-known large language models, as an example. GPT-3 has:

- **175 billion parameters** in total, which include weights and biases spread across numerous layers.
- **Multiple layers** of neurons (96 layers) with weights connecting neurons in adjacent layers.
- **Activation functions** (like GeLU - Gaussian Error Linear Unit) applied after the weighted sum of inputs to introduce non-linearity.
- **Dropout layers** to prevent overfitting during training.

# LLM Architectures
## Encoders and Decoders
Multiple architectures focused on encoding and decoding, i.e., embedding and text generation.

All the models are built on the Transformer Architecture ---- https://arxiv.org/abs/1706.03762 paper

Encoder don't need as much parameters as Decoders to perform well.

In the context of language models, encoding and decoding refer to two main tasks:

Encoding: This is the process of transforming input text into a format (often called embeddings) that the model can understand and work with.
Decoding: This is the process of generating text from the encoded information.
Embedding and Text Generation
Embedding: When text is converted into a numerical form (vectors) that can be processed by the model, this is called embedding. For example, the word "cat" might be represented as a vector like [0.25, -0.36, ...] in a high-dimensional space.
Text Generation: This involves producing human-like text based on the input. For example, given a prompt, the model can generate a continuation of the text.
Transformer Architecture
Transformer is a specific architecture for building models that handle tasks like encoding and decoding. Here's a breakdown of its components:

Attention Mechanism:

Definition: A system that allows the model to focus on different parts of the input text when making predictions.
Function: It helps the model understand the context by weighing the importance of different words in the sentence. For example, in the sentence "The cat sat on the mat," the word "sat" is closely related to "cat."
Encoder:

Function: Processes the input text and converts it into embeddings. It reads the entire input and uses the attention mechanism to understand the context and relationships between words.
Structure: Multiple layers of attention and feed-forward neural networks.
Decoder:

Function: Generates the output text using the embeddings produced by the encoder. It also uses the attention mechanism to understand the context and predict the next word.
Structure: Similar to the encoder, with additional layers that help in generating text sequentially.
Real-Life Example: Google Translate
Let's use Google Translate as an example:

Encoding: When you input a sentence in English, the model encodes the sentence into embeddings. This means it converts the English words into numerical vectors that capture their meanings and relationships.
Decoding: The model then decodes these embeddings to generate a sentence in the target language, like Spanish. It uses the context and meaning captured in the embeddings to produce a fluent and accurate translation.
Why Use Transformer Architecture?
Efficiency: Transformers handle long-range dependencies better than older models like RNNs (Recurrent Neural Networks). This means they can understand the context of words in long sentences more effectively.
Parallelization: Transformers allow for parallel processing, making training and inference faster.
Versatility: They can be used for various tasks such as translation, text generation, summarization, and more.
Example in Detail
Imagine you want to translate the sentence "I love programming" into French:

Encoding:

The words "I," "love," and "programming" are converted into embeddings.
The attention mechanism helps the model understand that "I" is the subject, "love" is the verb, and "programming" is the object.
Decoding:

Using the embeddings, the model generates the French sentence "J'adore la programmation."
The attention mechanism ensures that "J'" correctly matches "I" and "adore" matches "love," maintaining the correct context and meaning.
In summary, models built on the Transformer architecture are powerful tools for tasks that involve encoding input text into meaningful embeddings and decoding those embeddings to generate output text. This architecture is highly efficient and versatile, making it a popular choice for many natural language processing applications.

## Encoders
Models that convert a sequence of words to an embedding (vector representation)
### Examples:
* MiniLM
* Embed-light
* BERT
* RoBERTa
* DistillBERT
* SBERT

## Decoders
Models take a sequence of words and output next word
### Examples
* GPT-4
* Llama
* BLOOM
* Falcon

## Encoders - Decoders
Encodes a sequence of words and use the encoding to output the next word
### Examples
* T5
* UL2
* BART

## Tasks historically performed
| Task | Encoders | Decoders | Encoder-Decoder |
|-|-|-|-|
| Embedding text            | Yes | No | No |
| Abstractive QA            | No | Yes | Yes |
| Extractive QA             | Yes | Maybe | Yes |
| Translation               | No | Maybe | Yes |
| Creative writing          | No | Yes | No |
| Abstractive Summarization | No | Yes | Yes |
| Extractive Summarization  | Yes | Maybe | Yes |
| Chat                      | No | Yes | No |
| Forecasting               | No | No | No |
| Code                      | No | Yes | Yes |

# Prompting and Prompt Engineering
To exert some control over the LLM, we can affect the probability over vocabulary in ways

## Prompting
The simpliest way to affect the distribution over the vocabularity is to change the prompt

## Prompt
The text provided to an LLM as input, sometimes containing instructions and/or examples

## Prompt engineering
The process of iteratively refining a prompt for the purpose of eliciting a particular style of response
> Not guaranteed to work

> Although good prompts can result in better answers

## In-context Learning and Few-shot Prompting
### In-context learning
Conditioning (prompting) an LLM with instructions and demonstrations of the task it is meant to complete

### K-shot prompting
Explicitly providing _k_ examples of the intended task in the prompt

## Advanced Prompting Strategies
### Chain-of-Thought
Prompt the LLM to emit intermediate reasoning steps

### Least-to-Most
Prompt the LLM to decompose the problem and solve, easy-first

### Step-Back
Prompt the LLM to identify high-level concepts pertinent to a specific task

## Issues with prompting
### Prompt Injection (Jailbrake)
To deliberately provide an LLM with input that attempts to cause it to ignore instructions, cause harm, or behave contrary to deployment expectation

https://arxiv.org/abs/2306.05499

### Memorization
After answering, repeat the original prompt
* Leaked prompts
* Leakead private information from training

# Training
Prompting alone may be inappropriate when: training data exists, or domain adaption is required.

## Domain-adaption
Adapting a model (typically via training) to enhance its performance _outside_ of the domain/subject-area it was trained on.

## Training Syles
| Training Style | Modifies | Data | Summary |
|-|-|-|-|
| Fine-tuning (FT) | All parameters | Labeled, task-specific | Classic ML training |
| Param. Efficient FT | Few, new parameters | Labeled, task-specific | Learnable params to LLM |
| Soft prompting | Few, new parameters | Labeled, task-specific | Learnable prompt params |
| (cont.) pre-training | All parameters | unlabeled | Same as LLM pre-training |

# Decoding
The process of generating text with an LLM
* Decoding happens iteratively, 1 word at a time
* At each step of decoding, we use the distribution over vocabulary and select 1 word to emit
* The word is appended to the input, the decoding process continues

## Greedy Decoding
Pick the highest probability word at each step

## Non-Deterministic Decoding
Pick randomly among high probability candidates at each step

## Temperature
When decoding _temperature_ is a (hyper) parameter that modulates the distribution over vocabulary

* When temperature is **decreased**, the distribution is more _peaked_ around the most likely word
* When temperature is **increased**, the distribution is more _flattened_ over all words
* With sampling on, increasing the temperature makes the model deviate more from greedy decoding

> The realative ordering of the words is unaffected by temperature

# Hallucination
Generated text that is non-factual and/or ungrounded. This text often sounds logical and sensible.
* There are some methods that are claimed to reduce hallucination (e.g., retrieval-augmentation)
* There is no kown methodology to reliably keep LLMs from hallucinating

## Groundness and Attributability
### Grounded
Generated text is _grounded_ in a document if the document supports the text
* The research community has embraced attribution/grounding
* Attributed QA, system must ouput a document that grounds its answer
* The **TRUE** model: for measuring groundedness via NLI
* Train an LLM to output sentences _with citations_

# LLM Applications

## Retrieval Augmented Generation
* Primarily used in QA, where the model has access to (retrieved) support documents for a query
* Claimed to reduce hallucination
* Multi-document QA via fancy decoding, e.g., RAG-tok
* Idea has gotten a lot of traction
    * Used in dialogue, QA, fact-checking, slot filling, entity-linking
    * Non-parametric; in theory, the same model can answer questions about any corpus
    * Can be trained end-to-end

## Code Models
* Instead of training on written language, train on code and comments
* Co-pilot, Codex, Code Llama
* Complete partly written functions, synthesize programs from docstrings, debugging
* Largely successful: >85% of people using Co-pilot feel more productive
* Great fit between training data (code + comments) and test-time tasks (write code + comments). Also, is structured -> easier to learn

This is unlike LLMs, which are trained on a wide variety of internet text and used for many purposes (other than generating internet text); code models have (arguably) narrower scope

## Multi-Modal
* These are models trained on multiple modalities, e.g., language and images
* Models can be autoregressive, e.g., DALL-E or diffusion-based e.g., Stable Diffucion
* Diffusion-models can produce a complex output simultaneously, rather than token-by-token
    * Difficult to apply text because text is categorical
    * Some attempts have been made; still not very popular
* These models can perform either image-to-text, text-to-image tasks (or both), video generation, audio generation
* Recent retrieval-aumentation extensions

## Language Agents
* A building area of reseach where LLM-based _agents_
    * Create plans and "reason"
    * Take actions in response to plans and the environment
    * Are capable of using tools
* Some notable work in this space:
    * ReAct: Iterative framework where LLM emits _thoughts_, then _acts_, and _observes_ result
    * Toolformer: Pre-training technique where strings are replaced with calls to tools that yield result
    * Bootstrapped reasoning: Prompt the LLM to emit rationalization of intermediate steps; use as fine-tuning data
