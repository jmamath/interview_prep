# OpenAI Interview Preparation Textbook

## Current Status: Chapters 1 & 2 Complete ✅

### Overview

This textbook is designed to prepare candidates for OpenAI machine learning and coding interviews. It combines foundational CS concepts with practical ML engineering knowledge through a hands-on, exercise-driven approach.

### Pedagogical Philosophy

- **Exercise-driven learning**: Each chapter introduces concepts through real-world problems, not disconnected bullet points
- **Progressive difficulty**: Problems within each chapter flow from easy to hard
- **Practical context**: Every concept is tied directly to an exercise and includes relevant examples
- **Focused content**: Only key concepts necessary to solve exercises are included; no noise or filler
- **Student-centered**: Starter code with TODOs guides students, not complete solutions

### Structure per Chapter

Each chapter contains:
- **5 problems** progressing from easy to hard
- **2-3 focused tests** per problem (comprehensive validation)
- **Contextual introduction** with real-world examples
- **Key concepts** directly tied to exercise requirements
- **Starter code with TODOs** (not complete solutions!)
- **Example code** demonstrating concepts without giving away solutions
- **Hidden hints** (collapsible markdown sections)
- **2-4 pages** of moderate-depth refresher content (only essential concepts)

---

## Part 1: Core Coding Foundations

### ✅ Chapter 1: Data Structures & Complexity Analysis

**Status**: COMPLETE

**Learning Objectives:**
- Analyze algorithm complexity (Big O time and space)
- Understand memory layout, cache effects, and vectorization
- Implement memory-efficient algorithms
- Profile and optimize memory usage

**Problems & Structure:**

1. **Memory-Efficient Prefix Sum** (Easy)
   - Contextual Introduction: Sequence modeling use case
   - Key Concepts: In-place operations, space complexity
   - Example: Shows input/output transformation
   - Starter Code: 1 TODO for student implementation
   - Tests: 4 test cases for verification
   - Hints: Collapsible hint section

2. **Tensor Operation Benchmarking** (Easy-Medium)
   - Contextual Introduction: Vectorization importance in ML
   - Key Concepts: SIMD, benchmarking, performance measurement
   - Example: Loop vs vectorized comparison with explanations
   - Starter Code: 7 TODOs for full implementation
   - Tests: 3 test cases
   - Hints: Guidance on timing and plotting

3. **Cache-Aware Matrix Operations** (Medium)
   - Contextual Introduction: Cache hierarchy and locality
   - Key Concepts: Tiling, spatial/temporal locality, Numba JIT
   - Example: 2x2 matrix showing naive vs tiled approach with results
   - Starter Code: 4 TODOs for implementations
   - Tests: 3 test cases
   - Hints: Explanation of tile structure

4. **Custom Sparse Tensor Operations** (Medium-Hard)
   - Contextual Introduction: Netflix recommendation scenario
   - Key Concepts: COO format, sparsity, memory efficiency
   - Example: 4x8 user-item matrix showing dense vs sparse with calculations
   - Starter Code: 4 TODOs for SparseTensor class
   - Tests: 3 test cases
   - Hints: COO format guidance

5. **Memory Profiling and Optimization** (Hard)
   - Contextual Introduction: Large model training constraints
   - Key Concepts: Memory profiling, gradient checkpointing, activation recomputation
   - Example: Layer-by-layer breakdown of checkpointing (12.8MB → 1.3MB)
   - Starter Code: 5 TODOs for profiler and optimization
   - Tests: 3 test cases
   - Hints: Memory tracking guidance

**Implementation Details:**
- Total TODOs: 21 across all 5 problems
- All code is **starter code**, not complete solutions
- Each problem has working examples that students can run
- Tests verify correctness without revealing implementation details

---

### ✅ Chapter 2: Classic DP/Graphs for ML Engineers

**Status**: COMPLETE

**Learning Objectives:**
- Implement beam search variants for sequence generation
- Understand Hidden Markov Models and the Viterbi algorithm
- Apply constraints during generation
- Balance quality and diversity in outputs

**Problems & Structure:**

1. **Simple Beam Search** (Easy)
   - Context: Machine translation use case
   - Key Concepts: Beam width, greedy vs beam search
   - Example: Step-by-step beam search walkthrough with small vocabulary
   - Exercise: Implement basic `BeamSearch` class
   - Tests: 3 test cases

2. **Top-k Beam Search with Scores** (Medium)
   - Context: Summary generation diversity
   - Key Concepts: Length normalization, diversity penalty, temperature sampling
   - Example: Length normalization vs diversity penalty calculations
   - Exercise: Extend to `TopKBeamSearch` with scoring
   - Tests: 3 test cases

3. **Viterbi Algorithm for Sequence Tagging** (Medium)
   - Context: POS tagging in NLP
   - Key Concepts: HMMs, transition/emission probabilities, forward/backward passes
   - Example: Manual Viterbi computation trace for "the cat sat"
   - Exercise: Implement `HMMTagger` with Viterbi algorithm
   - Tests: 3 test cases

4. **Constrained Beam Search** (Medium-Hard)
   - Context: Chatbot safety constraints
   - Key Concepts: Constraint satisfaction, early pruning
   - Example: How constraints filter beam candidates and prune search space
   - Exercise: Implement constraint classes and `ConstrainedBeamSearch`
   - Tests: 3 test cases

5. **Diverse Beam Search with Groups** (Hard)
   - Context: Creative writing diversity
   - Key Concepts: Sequence similarity, grouping, quality-diversity tradeoff
   - Example: Jaccard similarity calculation and grouping for diversity
   - Exercise: Implement `DiverseBeamSearch` with similarity-based grouping
   - Tests: 3 test cases

**Implementation Details:**
- All problems have contextual introductions with real-world ML applications
- All problems include example code demonstrating concepts without revealing solutions
- Examples cover: Beam search steps, length normalization, Viterbi computation, constraint filtering, sequence similarity
- Starter code with TODO markers (not complete solutions)
- 2-3 focused tests per questi for validation
- JSON notebook is valid and parses correctly

---

### ✅ Chapter 3: Streaming & Online Algorithms

**Status**: COMPLETE

**Learning Objectives:**
- Process data that doesn't fit in memory (streaming)
- Implement online learning algorithms
- Understand frequency estimation and cardinality counting
- Apply streaming algorithms to real-time ML systems

**Problems & Structure:**

1. **Frequency Estimation with Count-Min Sketch** (Easy-Medium)
   - Context: Google search query frequency estimation
   - Key Concepts: Probabilistic data structures, hash collisions, space-time tradeoff
   - Example: Step-by-step sketch updates with 2 rows and 4 buckets
   - Starter Code: TODO for update() and query() methods
   - Tests: 3 test cases

2. **Cardinality Estimation with HyperLogLog** (Medium)
   - Context: Facebook unique daily active user counting
   - Key Concepts: Probabilistic counting, harmonic mean, leading zero patterns
   - Example: Coin flip intuition and register updates with binary hashing
   - Starter Code: TODO for add() and count() methods
   - Tests: 2 test cases

3. **Online Mean and Variance Computation** (Medium)
   - Context: BatchNorm statistics for neural networks
   - Key Concepts: Numerical stability, Welford's algorithm
   - Example: Complete trace of mean/variance computation for [4,7,13,16]
   - Starter Code: TODO for update() and variance() methods
   - Tests: 2 test cases

4. **Reservoir Sampling** (Medium)
   - Context: Recommendation system sampling from activity streams
   - Key Concepts: Uniform random sampling, single-pass algorithm
   - Example: 8-step walkthrough showing probability preservation
   - Starter Code: TODO for add() method
   - Tests: 2 test cases

5. **Online Gradient Descent** (Hard)
   - Context: Real-time learning for online advertising
   - Key Concepts: SGD, learning rate schedules, regret bounds
   - Example: 2 gradient steps showing weight updates and lr decay
   - Starter Code: TODO for partial_fit() and predict() methods
   - Tests: 2 test cases

**Implementation Details:**
- All problems have contextual introductions with real-world ML applications
- All problems include example code demonstrating algorithms without revealing solutions
- Examples cover: Hash table mechanics, probabilistic counting, numerical stability, sampling proofs, gradient updates
- Starter code with TODO markers (not complete solutions)
- 2-3 focused tests per question for validation
- JSON notebook is valid and parses correctly

---

### Chapter 4: Optimization Algorithms

**Status**: NOT STARTED

**Learning Objectives:**
- Understand gradient-based optimization methods
- Implement momentum, adaptive learning rates, and second-order methods
- Analyze convergence properties

**Planned Problems:**
1. Gradient Descent Variants (Easy-Medium)
2. Adam Optimizer (Medium)
3. Learning Rate Scheduling (Medium)
4. Newton's Method and Quasi-Newton (Medium-Hard)
5. Constrained Optimization (Hard)

---

## Part 2: ML Engineering Essentials

### Chapter 5: Training Efficiency & Memory

**Status**: NOT STARTED

**Learning Objectives:**
- Implement gradient checkpointing and mixed-precision training
- Understand batch size effects on convergence
- Optimize memory usage during backpropagation

**Planned Problems:**
1. Gradient Accumulation (Easy-Medium)
2. Gradient Checkpointing (Medium)
3. Mixed-Precision Training (Medium)
4. Data Loading and Batching (Medium-Hard)
5. Advanced Memory Optimization (Hard)

---

### Chapter 6: Inference Efficiency & Quantization

**Status**: NOT STARTED

**Learning Objectives:**
- Implement model quantization (INT8, dynamic quantization)
- Optimize inference through pruning and distillation
- Understand latency vs accuracy trade-offs

**Planned Problems:**
1. Post-Training Quantization (Easy-Medium)
2. Quantization-Aware Training (Medium)
3. Pruning Strategies (Medium)
4. Knowledge Distillation (Medium-Hard)
5. Inference Optimization Pipeline (Hard)

---

### Chapter 7: Dataset Processing & Shuffling

**Status**: NOT STARTED

**Learning Objectives:**
- Implement efficient data shuffling algorithms
- Understand data preprocessing pipelines
- Apply augmentation strategies

**Planned Problems:**
1. Efficient Shuffling Algorithms (Easy-Medium)
2. Data Normalization and Standardization (Medium)
3. Data Augmentation (Medium)
4. Handling Imbalanced Data (Medium-Hard)
5. Advanced Data Pipelines (Hard)

---

## Part 3: Applied LLM & Pretraining Context

### Chapter 8: Loss Functions & Scaling Laws

**Status**: NOT STARTED

**Learning Objectives:**
- Understand various loss functions and when to use them
- Understand scaling laws and their implications
- Implement custom loss functions

**Planned Problems:**
1. Cross-Entropy and Variants (Easy-Medium)
2. Contrastive Learning Losses (Medium)
3. Ranking and Margin Losses (Medium)
4. Custom Loss Functions (Medium-Hard)
5. Scaling Laws and Compute Optimization (Hard)

---

### Chapter 9: Debugging & Sanity Checking

**Status**: NOT STARTED

**Learning Objectives:**
- Implement debugging techniques for ML models
- Perform sanity checks to catch common errors
- Analyze model behavior and failure modes

**Planned Problems:**
1. Gradient Checking (Easy-Medium)
2. Activation Statistics Monitoring (Medium)
3. Loss Curve Analysis (Medium)
4. Weight and Gradient Analysis (Medium-Hard)
5. Comprehensive Sanity Check Suite (Hard)

---

## Implementation Guidelines

### Code Quality Standards
- Type hints on all functions
- Comprehensive docstrings
- Clear comments for non-obvious logic
- Minimal code; prefer clarity over cleverness
- Starter code with TODO markers, not complete solutions

### Testing Philosophy
- 2-3 focused tests per problem
- Tests cover core functionality, edge cases, and practical scenarios
- Performance/benchmarking tests where relevant
- All tests should pass without external dependencies

### Documentation Standards
- Contextual introductions with real-world examples
- Key concepts explained only as needed for the exercise
- Clear problem statements and requirements
- **Example code** demonstrating concepts without revealing solutions
- Practical examples with expected outputs

### Notebook Format
- Open in Colab button at top
- Exercise-driven layout (not topical refresher then exercises)
- Contextual examples before starter code
- Hints in collapsible markdown sections
- Test output with clear pass/fail indicators

### Exercise Structure
Each problem has this pattern:
```
1. Problem Description (Markdown)
   - Contextual Introduction
   - Key Concepts
   - Problem Statement
   - Requirements

2. Example Code (Code Cell)
   - Shows concepts in action
   - Demonstrates expected behavior
   - Does NOT give away solution

3. Starter Code (Code Cell)
   - Function signatures with docstrings
   - TODO markers for student implementation
   - Test function calls

4. Hints (Collapsible Markdown)
   - Conceptual guidance
   - Key algorithm insight
   - Common pitfall warnings
```

---

## Progression and Validation

✅ = Complete and validated
🔄 = In progress
⏳ = Planned

1. **Chapter 1**: Data Structures & Complexity ✅
   - All 5 problems with exercises
   - 21 TODOs total
   - Examples for all problems
   - JSON validated

2. **Chapter 2**: Classic DP/Graphs ✅
    - All 5 problems with contextual introductions
    - Ready for exercise conversion (same pattern as Ch1)
    - Comprehensive tests included

3. **Chapter 3**: Streaming & Online Algorithms ✅
   - All 5 problems with contextual introductions
   - All 5 problems now include example code demonstrations
   - Starter code structure ready for student exercises
   - Comprehensive tests included (2-3 per problem)

4. **Chapter 4**: Optimization Algorithms ⏳
5. **Chapter 5**: Training Efficiency & Memory ⏳
6. **Chapter 6**: Inference Efficiency & Quantization ⏳
7. **Chapter 7**: Dataset Processing & Shuffling ⏳
8. **Chapter 8**: Loss Functions & Scaling Laws ⏳
9. **Chapter 9**: Debugging & Sanity Checking ⏳

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

---

## Quality Checklist

For each completed chapter:
- [ ] All 5 problems have contextual introductions
- [ ] All problems have real-world examples or use cases
- [ ] All problems have example code demonstrating concepts
- [ ] All code cells are starter code with TODOs (not solutions)
- [ ] Each problem has 2-3 focused tests
- [ ] All tests pass when TODOs are properly implemented
- [ ] All hints are collapsible and provide guidance, not solutions
- [ ] JSON notebook is valid and parses correctly
- [ ] Chapter is reviewed for pedagogy and clarity
