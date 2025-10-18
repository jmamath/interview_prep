# OpenAI Interview Preparation Textbook

## Overview

This textbook is designed to prepare candidates for OpenAI machine learning and coding interviews. It combines foundational CS concepts with practical ML engineering knowledge through a hands-on, exercise-driven approach.

### Pedagogical Philosophy

- **Exercise-driven learning**: Each chapter introduces concepts through real-world problems, not disconnected bullet points
- **Progressive difficulty**: Problems within each chapter flow from easy to hard
- **Practical context**: Every concept is tied directly to an exercise and includes relevant examples
- **Focused content**: Only key concepts necessary to solve exercises are included; no noise or filler

### Structure per Chapter

Each chapter contains:
- **5 problems** progressing from easy to hard
- **2-3 focused tests** per problem (comprehensive validation)
- **Contextual introduction** with real-world examples
- **Key concepts** directly tied to exercise requirements
- **Starter code** with TODO placeholders for implementation
- **Hidden hints** (collapsible markdown sections)
- **2-4 pages** of moderate-depth refresher content (only essential concepts)

---

## Part 1: Core Coding Foundations

### Chapter 1: Data Structures & Complexity Analysis

**Learning Objectives:**
- Analyze algorithm complexity (Big O time and space)
- Understand memory layout, cache effects, and vectorization
- Implement memory-efficient algorithms
- Profile and optimize memory usage

**Problems:**

1. **Memory-Efficient Prefix Sum** (Easy)
   - Context: Prefix sums are fundamental for range queries; memory-efficient in-place algorithms matter
   - Key Concepts: In-place operations, space complexity
   - Output: `prefix_sum_inplace(arr)` function with in-place modification verification

2. **Tensor Operation Benchmarking** (Easy-Medium)
   - Context: Understanding vectorization vs loops is crucial for ML performance
   - Key Concepts: Vectorization, loop optimization, benchmarking
   - Output: Comparison of elementwise operations with performance analysis

3. **Cache-Aware Matrix Operations** (Medium)
   - Context: Cache locality affects real-world performance significantly
   - Key Concepts: Cache effects, tiling strategies, Numba compilation
   - Output: Multiple matrix multiplication implementations with tiling and optimization

4. **Custom Sparse Tensor Operations** (Medium-Hard)
   - Context: Many ML applications involve sparse data (embeddings, graphs)
   - Key Concepts: Sparse representations, efficient operations, memory usage
   - Output: `SparseTensor` class with operations and memory profiling

5. **Memory Profiling and Optimization** (Hard)
   - Context: Large neural networks require careful memory management during training
   - Key Concepts: Memory profiling tools, gradient checkpointing, optimization strategies
   - Output: `MemoryProfiler` class and comparison of memory strategies

---

### Chapter 2: Classic DP/Graphs for ML Engineers

**Learning Objectives:**
- Implement beam search variants for sequence generation
- Understand Hidden Markov Models and the Viterbi algorithm
- Apply constraints during generation
- Balance quality and diversity in outputs

**Problems:**

1. **Simple Beam Search** (Easy)
   - Context: Machine translation uses beam search to generate target sequences
   - Key Concepts: Beam width, greedy search vs beam search
   - Output: `BeamSearch` class for basic sequence generation

2. **Top-k Beam Search with Scores** (Medium)
   - Context: Summary generation requires diverse content; top-k variations help maintain diversity
   - Key Concepts: Length normalization, diversity penalty, temperature sampling
   - Output: `TopKBeamSearch` class with scoring and diversity measures

3. **Viterbi Algorithm for Sequence Tagging** (Medium)
   - Context: POS tagging and NER in NLP rely on Hidden Markov Models
   - Key Concepts: Transition probabilities, emission probabilities, forward/backward passes
   - Output: `HMMTagger` class implementing Viterbi algorithm

4. **Constrained Beam Search** (Medium-Hard)
   - Context: Chatbots need to generate text respecting safety constraints
   - Key Concepts: Constraint satisfaction, early pruning, fallback strategies
   - Output: Constraint classes and `ConstrainedBeamSearch` implementation

5. **Diverse Beam Search with Groups** (Hard)
   - Context: Creative writing applications benefit from diverse generation strategies
   - Key Concepts: Sequence similarity, clustering/grouping, quality-diversity tradeoff
   - Output: `DiverseBeamSearch` class with grouping and selection mechanisms

---

### Chapter 3: Streaming & Online Algorithms

**Learning Objectives:**
- Process data that doesn't fit in memory (streaming)
- Implement online learning algorithms
- Understand frequency estimation and cardinality counting
- Apply these in ML contexts (mini-batch training, incremental model updates)

**Problems:**

1. **Frequency Estimation with Count-Min Sketch** (Easy-Medium)
   - Context: Identifying popular items in streaming data (trending topics, popular users)
   - Key Concepts: Space-efficient frequency tracking, hash collisions, approximation
   - Output: `CountMinSketch` data structure with query and update

2. **Cardinality Estimation with HyperLogLog** (Medium)
   - Context: Estimating unique users/items in a stream (analytics pipelines)
   - Key Concepts: Probabilistic counting, logarithmic space, accuracy tuning
   - Output: `HyperLogLog` implementation for approximate cardinality

3. **Online Mean and Variance Computation** (Medium)
   - Context: Computing statistics without storing all data (real-time metrics)
   - Key Concepts: Welford's algorithm, numerical stability, single-pass updates
   - Output: Online statistics calculator class

4. **Reservoir Sampling** (Medium)
   - Context: Selecting random samples from unlimited streams
   - Key Concepts: Uniform random sampling, memory-bounded buffers
   - Output: `ReservoirSampler` for k-samples from stream

5. **Online Gradient Descent** (Hard)
   - Context: Training models on streaming data (continual learning)
   - Key Concepts: Mini-batch SGD, learning rate scheduling, convergence properties
   - Output: `OnlineGradientDescent` optimizer with convergence analysis

---

### Chapter 4: Optimization Algorithms

**Learning Objectives:**
- Understand gradient-based optimization methods
- Implement momentum, adaptive learning rates, and second-order methods
- Analyze convergence properties
- Understand practical considerations in ML training

**Problems:**

1. **Gradient Descent Variants** (Easy-Medium)
   - Context: SGD, momentum, and Nesterov are fundamental to all model training
   - Key Concepts: Learning rates, momentum accumulation, convergence rates
   - Output: Multiple optimizer implementations with comparison

2. **Adam Optimizer** (Medium)
   - Context: Adam is widely used in practice due to adaptive learning rates
   - Key Concepts: First/second moment estimation, bias correction, practical hyperparameters
   - Output: `AdamOptimizer` implementation with convergence properties

3. **Learning Rate Scheduling** (Medium)
   - Context: Scheduling learning rates improves convergence and generalization
   - Key Concepts: Warmup, decay schedules, cyclical learning rates
   - Output: Multiple scheduler implementations and their effects

4. **Newton's Method and Quasi-Newton** (Medium-Hard)
   - Context: Second-order methods for faster convergence in certain regimes
   - Key Concepts: Hessian approximation, L-BFGS, line search
   - Output: Implementation and comparison with first-order methods

5. **Constrained Optimization** (Hard)
   - Context: Many real problems have constraints (e.g., pruning, knowledge distillation)
   - Key Concepts: Lagrange multipliers, penalty methods, projected gradients
   - Output: Solvers for constrained optimization problems

---

## Part 2: ML Engineering Essentials

### Chapter 5: Training Efficiency & Memory

**Learning Objectives:**
- Implement gradient checkpointing and mixed-precision training
- Understand batch size effects on convergence
- Optimize memory usage during backpropagation
- Profile and improve training efficiency

**Problems:**

1. **Gradient Accumulation** (Easy-Medium)
   - Context: Simulating larger batch sizes on limited hardware
   - Key Concepts: Gradient scaling, effective batch size, training stability
   - Output: Gradient accumulation mechanism with loss tracking

2. **Gradient Checkpointing** (Medium)
   - Context: Reducing memory for deep networks by recomputing activations
   - Key Concepts: Activation storage, forward/backward trade-offs, implementation
   - Output: Checkpointing wrapper for layer sequences

3. **Mixed-Precision Training** (Medium)
   - Context: Using FP16 for speed and memory with FP32 for stability
   - Key Concepts: Numeric precision, loss scaling, gradient underflow
   - Output: Mixed-precision training loop with appropriate scaling

4. **Data Loading and Batching** (Medium-Hard)
   - Context: Efficient data pipelines are critical for training speed
   - Key Concepts: Prefetching, caching, dynamic batching, multi-worker loading
   - Output: Custom data loader with performance profiling

5. **Advanced Memory Optimization** (Hard)
   - Context: Techniques used in production training of large models
   - Key Concepts: Activation recomputation scheduling, parameter sharding, zero-copy operations
   - Output: Comprehensive memory optimization combining multiple techniques

---

### Chapter 6: Inference Efficiency & Quantization

**Learning Objectives:**
- Implement model quantization (INT8, dynamic quantization)
- Optimize inference through pruning and distillation
- Understand latency vs accuracy trade-offs
- Profile and improve inference speed

**Problems:**

1. **Post-Training Quantization** (Easy-Medium)
   - Context: Reducing model size for deployment without retraining
   - Key Concepts: Calibration, scale factors, INT8 representation
   - Output: INT8 quantization with calibration

2. **Quantization-Aware Training** (Medium)
   - Context: Training with quantization awareness for better accuracy
   - Key Concepts: Fake quantization, simulated quantization, gradient flow
   - Output: QAT implementation with training loop

3. **Pruning Strategies** (Medium)
   - Context: Removing unnecessary parameters reduces computation
   - Key Concepts: Magnitude pruning, structured pruning, sparsity
   - Output: Pruning algorithms and sparsity analysis

4. **Knowledge Distillation** (Medium-Hard)
   - Context: Compressing knowledge into smaller, faster models
   - Key Concepts: Temperature scaling, soft targets, loss weighting
   - Output: Distillation loss and training procedure

5. **Inference Optimization Pipeline** (Hard)
   - Context: Combining multiple techniques for production inference
   - Key Concepts: Model fusion, graph optimization, kernel selection, batching
   - Output: End-to-end inference optimization combining quantization, pruning, distillation

---

### Chapter 7: Dataset Processing & Shuffling

**Learning Objectives:**
- Implement efficient data shuffling algorithms
- Understand data preprocessing pipelines
- Apply augmentation strategies
- Handle imbalanced and skewed datasets

**Problems:**

1. **Efficient Shuffling Algorithms** (Easy-Medium)
   - Context: Large datasets require shuffle strategies that don't load everything into memory
   - Key Concepts: Fisher-Yates, stratified sampling, shuffle buffers
   - Output: Multiple shuffling implementations with memory/speed trade-offs

2. **Data Normalization and Standardization** (Medium)
   - Context: Proper normalization significantly affects model training
   - Key Concepts: Z-score normalization, whitening, running statistics
   - Output: Normalization implementations with handling of edge cases

3. **Data Augmentation** (Medium)
   - Context: Augmentation improves generalization and data efficiency
   - Key Concepts: Transformations, composition, probabilistic application
   - Output: Augmentation pipeline for images and text

4. **Handling Imbalanced Data** (Medium-Hard)
   - Context: Many real datasets have class imbalance
   - Key Concepts: Oversampling, undersampling, SMOTE, weighted losses
   - Output: Resampling strategies and loss weighting

5. **Advanced Data Pipelines** (Hard)
   - Context: Production data pipelines require handling multiple formats and scales
   - Key Concepts: Streaming, validation, caching, multi-format support
   - Output: Robust pipeline with validation and error handling

---

## Part 3: Applied LLM & Pretraining Context

### Chapter 8: Loss Functions & Scaling Laws

**Learning Objectives:**
- Understand various loss functions and when to use them
- Understand scaling laws and their implications
- Implement custom loss functions
- Analyze loss behavior and training dynamics

**Problems:**

1. **Cross-Entropy and Variants** (Easy-Medium)
   - Context: Cross-entropy is foundational; variants handle different scenarios
   - Key Concepts: Categorical/binary cross-entropy, label smoothing, focal loss
   - Output: Multiple loss function implementations with numerical stability

2. **Contrastive Learning Losses** (Medium)
   - Context: Self-supervised learning uses contrastive objectives
   - Key Concepts: Similarity metrics, temperature, mining strategies
   - Output: SimCLR and triplet loss implementations

3. **Ranking and Margin Losses** (Medium)
   - Context: Ranking problems (recommendation, retrieval) use specialized losses
   - Key Concepts: Margin-based losses, hinge loss, ranking metrics
   - Output: Ranking loss implementations with metric analysis

4. **Custom Loss Functions** (Medium-Hard)
   - Context: Domain-specific problems often require custom losses
   - Key Concepts: Gradient computation, numerical stability, weighting schemes
   - Output: Implementation of domain-specific loss with examples

5. **Scaling Laws and Compute Optimization** (Hard)
   - Context: Understanding scaling laws guides training decisions
   - Key Concepts: Power laws, loss scaling, compute allocation, convergence prediction
   - Output: Loss scaling analysis and prediction models

---

### Chapter 9: Debugging & Sanity Checking

**Learning Objectives:**
- Implement debugging techniques for ML models
- Perform sanity checks to catch common errors
- Analyze model behavior and failure modes
- Use visualization and analysis tools effectively

**Problems:**

1. **Gradient Checking** (Easy-Medium)
   - Context: Verifying gradient correctness prevents subtle training bugs
   - Key Concepts: Numerical differentiation, relative error, finite differences
   - Output: Gradient checker with detailed error reporting

2. **Activation Statistics Monitoring** (Medium)
   - Context: Monitoring activations reveals training pathologies
   - Key Concepts: Mean/variance tracking, dead neurons, saturation detection
   - Output: Activation monitor with statistics and visualization

3. **Loss Curve Analysis** (Medium)
   - Context: Loss curves reveal many training issues
   - Key Concepts: Divergence detection, convergence verification, overfitting signals
   - Output: Loss analyzer with automatic issue detection

4. **Weight and Gradient Analysis** (Medium-Hard)
   - Context: Analyzing weight/gradient distributions catches training issues
   - Key Concepts: Initialization verification, gradient flow, dead weights
   - Output: Statistical analyzer for weights and gradients

5. **Comprehensive Sanity Check Suite** (Hard)
   - Context: Production training requires automated sanity checks
   - Key Concepts: Multiple check types, early termination criteria, logging
   - Output: Complete sanity check system combining all techniques

---

## Implementation Guidelines

### Code Quality Standards
- Type hints on all functions
- Comprehensive docstrings
- Clear comments for non-obvious logic
- Minimal code; prefer clarity over cleverness

### Testing Philosophy
- 2-3 focused tests per problem
- Tests cover core functionality, edge cases, and practical scenarios
- Performance/benchmarking tests where relevant
- All tests should pass without external dependencies

### Documentation Standards
- Contextual introductions with real-world examples
- Key concepts explained only as needed for the exercise
- Clear problem statements and requirements
- Practical examples demonstrating concepts

### Notebook Format
- Open in Colab button at top
- Exercise-driven layout (not topical refresher then exercises)
- Hints in collapsible markdown sections
- Test output with clear pass/fail indicators

---

## Progression and Validation

Each chapter will be completed and validated before moving to the next:

1. **Chapter 1**: Data Structures & Complexity ✅ (Completed)
2. **Chapter 2**: Classic DP/Graphs ✅ (Completed - pedagogical revision)
3. **Chapter 3**: Streaming & Online Algorithms (Next)
4. **Chapter 4**: Optimization Algorithms
5. **Chapter 5**: Training Efficiency & Memory
6. **Chapter 6**: Inference Efficiency & Quantization
7. **Chapter 7**: Dataset Processing & Shuffling
8. **Chapter 8**: Loss Functions & Scaling Laws
9. **Chapter 9**: Debugging & Sanity Checking

---

## Technology Stack

- **Python 3.8+**
- **NumPy**: Numerical computations
- **PyTorch**: ML framework and examples
- **Jupyter/Google Colab**: Notebook environment
- **Type hints**: For code clarity
- **Standard library**: collections, heapq, abc (for interfaces)

---

## Target Audience

This textbook is designed for:
- Mid-to-senior software engineers transitioning to ML
- ML engineers preparing for OpenAI-style interviews
- Students wanting practical ML engineering knowledge
- Anyone seeking to understand ML infrastructure and optimization

**Prerequisites:**
- Comfortable with Python and data structures
- Basic understanding of algorithms and complexity analysis
- Familiarity with machine learning concepts (loss functions, optimization)
