# Quantum Transformation Circuit (QTC)
This repository implements the integration of a fully-connected Quantum Transformation Circuit into the Capsule Network pipeline, following the work on the  [Hybrid Quantum Capsule Network (HQCapsNet)](https://github.com/RinRoya/quantum-transformation-circuit).

---

## ⚙️ Environment & Dependencies

- Python: 3.10.15  
- TensorFlow: 2.10.0  
- PennyLane: 0.39.0  

This setup supports GPU acceleration for classical components and statevector simulation via PennyLane `default.qubit`.

---

## 🚀 Model Overview

### HQCapsNet consists of:
- Convolutional feature extractor
- Primary Capsule Layer
- Quantum Transformation Capsule (QTC) layer
- Dynamic routing mechanism

### The QTC layer is implemented using:
- Amplitude / Angle / IQP embeddings
- Variational quantum ansatz (Basic / Strongly Entangling / Simplified Two-Design)
- Measurement via computational basis probabilities

### Noise Model (Optional)

A simplified NISQ-inspired noise model is included:
- Terminal depolarizing noise
- Symmetric readout error

Noise is applied directly on measurement probabilities as an efficient approximation of quantum hardware imperfections without requiring density-matrix simulation. If full density-matrix simulation with additional quantum channels is required, the implementation should use PennyLane `default.mixed`. However, such simulations significantly increase computational cost and memory usage. Therefore, the current implementation uses analytical probability-level noise injection for efficiency, and avoids enabling `tf.config.run_functions_eagerly(True)` or full mixed-state execution due to severe performance overhead.

---

## 🔧 Usage

### Classical Capsule Network Flow (Classical Version)

Input example: MNIST (28×28×1)

```Python
x = self.convolutional_layer(x) # (None, 20, 20, 256)
x = self.primary_capsule_layer(x) # (None, 6, 6, 256)
u = tf.reshape(x, (x.shape[0], -1, self.dim)) # (None, 576, 16)
u = tf.expand_dims(u, axis=-2) # (None, 576, 1, 16)
u = tf.expand_dims(u, axis=-1) # (None, 576, 1, 16, 1)
u_hat = tf.matmul(self.w, u) # (None, 576, 2, 8, 1)
u_hat = tf.squeeze(u_hat, [-1]) # (None, 576, 2, 8)
```
Dynamic routing procedure follows...


---

### Fully-connected QTC (Quantum Version)

```Python
self.QC_pqc = QuantumClass(n_qubits_pqc, n_pqc_layer, pqc_layer, q_embedding, imprimitive)
self.PQC_Layer = QuantumLayer(self.QC_pqc.qnode_pqc, self.QC_pqc.params_shape, num_class, depolarizing_prop, readout_prob)

x = self.convolutional_layer(x) # (None, 20, 20, 256)
x = self.primary_capsule_layer(x) # (None, 6, 6, 256)
u = tf.reshape(x, (x.shape[0], -1, self.dim)) # (None, 576, 16)
u_hat = self.PQC_Layer(u) # (None, 576, 2, 8)
u_hat = tf.cast(u_hat, dtype=tf.float32)
```
Dynamic routing procedure follows...

---

### HQCapsNet model implementation

A simple implementation of the HQCapsNet model is available in `main.ipynb`

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{HQCapsNet2026,
  title   = {Hybrid Quantum Capsule Network: Quantum Transformation Circuit as Vote Transformation in Image Classification},
  author  = {Wijaya, Ridho Nur Rohman and Setiyono, Budi and Sulistyaningrum, Dwi Ratna},
  journal = {IEEE Transactions on Quantum Engineering},
  year    = {2026},
  note    = {Under review}
}
```

---

## 📜 License

MIT License

Permission is hereby granted, free of charge, to use, modify, and distribute this software with proper attribution.

---
