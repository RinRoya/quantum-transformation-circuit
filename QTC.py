class QuantumClass:
    def __init__(self, n_qubits_pqc, n_pqc_layer, pqc_layer, q_embedding, imprimitive=None):
        self.n_qubits_pqc = n_qubits_pqc
        self.dev = qp.device("default.qubit", wires=n_qubits_pqc)
        self.q_embedding = q_embedding
        self.imprimitive = imprimitive
 
        if pqc_layer=="basic":
            self.pqcAnsatz = self.pqcBasic
            self.params_shape = {"parameters": qp.BasicEntanglerLayers.shape(n_pqc_layer, self.n_qubits_pqc)}
        elif pqc_layer=="strongly":
            self.pqcAnsatz = self.pqcStrongly
            self.params_shape = {"parameters": qp.StronglyEntanglingLayers.shape(n_pqc_layer, self.n_qubits_pqc)}
        elif pqc_layer=="simplified":
            self.pqcAnsatz = self.pqcSimplified
            sim_init, sim_params = qp.SimplifiedTwoDesign.shape(n_pqc_layer, self.n_qubits_pqc)
            self.params_shape = {"init_parameters": sim_init, "parameters": sim_params}
        elif pqc_layer=="basicXYZ": # custom
            self.pqcAnsatz = self.pqcBasicXYZ
            self.params_shape = {"parametersX": qp.BasicEntanglerLayers.shape(1, self.n_qubits_pqc), 
                                 "parametersY": qp.BasicEntanglerLayers.shape(1, self.n_qubits_pqc), 
                                 "parametersZ": qp.BasicEntanglerLayers.shape(1, self.n_qubits_pqc)}
        else:
            raise ValueError(f"Unsupported PQC type: {pqc_layer}")
            
        self.qnode_pqc = qp.QNode(self.pqcAnsatz, self.dev, diff_method="backprop", interface="tf")
        
    def embedding(self, inputs):
        if self.q_embedding=="amplitude":
            qp.templates.AmplitudeEmbedding(inputs, wires=range(self.n_qubits_pqc), pad_with=0.0, normalize=True)
        elif self.q_embedding=="angle":
            qp.templates.AngleEmbedding(inputs, wires=range(self.n_qubits_pqc))
        elif self.q_embedding[:-1]=="iqp":
            qp.templates.IQPEmbedding(inputs, wires=range(self.n_qubits_pqc), n_repeats=int(self.q_embedding[-1]))
        else:
            raise ValueError(f"Unsupported embedding: {self.q_embedding}")
            
    def pqcBasic(self, inputs, parameters):
        self.embedding(inputs)
        qp.BasicEntanglerLayers(weights=parameters, wires=range(self.n_qubits_pqc), rotation=self.imprimitive)
        outputs = [qp.probs(i) for i in range(self.n_qubits_pqc)]
        return outputs
        
    def pqcBasicXYZ(self, inputs, parametersX, parametersY, parametersZ):
        self.embedding(inputs)
        qp.BasicEntanglerLayers(weights=parametersX, wires=range(self.n_qubits_pqc), rotation=qp.RX)
        qp.BasicEntanglerLayers(weights=parametersY, wires=range(self.n_qubits_pqc), rotation=qp.RY)
        qp.BasicEntanglerLayers(weights=parametersZ, wires=range(self.n_qubits_pqc), rotation=qp.RZ)
        outputs = [qp.probs(i) for i in range(self.n_qubits_pqc)]
        return outputs 
        
    def pqcStrongly(self, inputs, parameters):
        self.embedding(inputs)
        qp.StronglyEntanglingLayers(weights=parameters, wires=range(self.n_qubits_pqc), imprimitive=self.imprimitive)
        outputs = [qp.probs(i) for i in range(self.n_qubits_pqc)]
        return outputs 
        
    def pqcSimplified(self, inputs, init_parameters, parameters):
        self.embedding(inputs)
        qp.SimplifiedTwoDesign(initial_layer_weights=init_parameters, weights=parameters, wires=range(self.n_qubits_pqc))
        outputs = [qp.probs(i) for i in range(self.n_qubits_pqc)]
        return outputs 

class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, circuit, weight_shape, num_class, depolarizing_prop=0.0, readout_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.circuit = circuit
        self.num_class = num_class
        self.weight_shape = weight_shape

        self.depolarizing_prop = tf.Variable(depolarizing_prop, trainable=False, dtype=tf.float32, name="gate_noise_probability")
        self.readout_prob = tf.Variable(readout_prob, trainable=False, dtype=tf.float32, name="readout_error_probability")
    
    def build(self, input_shape):
        num_caps = input_shape[0]
        has_batch_dim = len(input_shape) > 2
        if has_batch_dim:
            num_caps = input_shape[1]
        self.num_pqc = num_caps*self.num_class
        self.weights_q = []
        for name_shape in self.weight_shape:
            self.weights_q.append(tf.Variable(tf.keras.initializers.RandomNormal(stddev=0.1)(shape=(self.num_pqc,*self.weight_shape[name_shape])), 
                                              trainable=True, dtype=tf.float32))

    def apply_noise(self, probabilities):
        # terminal depolarizing noise (state noise), wxact equivalent from qp.DepolarizingChannel(p)
        p = tf.cast(self.depolarizing_prop, probabilities.dtype)
        probabilities = ((1.0 - 4.0 * p / 3.0) * probabilities + 2.0 * p / 3.0)
    
        # followed by symmetric readout error (measurement noise)
        r = tf.cast(self.readout_prob, probabilities.dtype)
        flipped_probabilities = tf.reverse(probabilities, axis=[-1])
        probabilities = ((1.0 - r) * probabilities + r * flipped_probabilities)
    
        return probabilities
    
    def call(self, inputs):
        batch_dims = tf.shape(inputs)[:-1]
        inputs = tf.expand_dims(inputs, axis=-2)
        inputs = tf.repeat(inputs, repeats=self.num_class, axis=-2)
        inputs = tf.reshape(inputs, (-1, self.num_pqc, inputs.shape[-1]))

        outputs = tf.vectorized_map(lambda x: self.circuit(x + 1e-7, *self.weights_q), inputs)   
        outputs = tf.transpose(outputs, perm=[1, 2, 0, 3])

        outputs = self.apply_noise(outputs)
        
        outputs = tf.reshape(outputs, (batch_dims[0],batch_dims[1],self.num_class,-1))
        return outputs