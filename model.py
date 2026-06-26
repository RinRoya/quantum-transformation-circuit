import numpy as np
import tensorflow as tf
import pennylane as qp
from tqdm import tqdm
from QTC import QuantumClass, QuantumLayer

class HybridQuantumCapsuleNetwork(tf.keras.Model):
    def __init__(self, num_conv_layer_kernels=256, num_primary_capsules=32, num_primary_capsule_dim=8, num_digitcaps=10, num_digitcaps_dim=16, r=3,
                 reconstruction=True,conv_kernel_size=[9,9], conv_strides=[1,1], primary_capsule_kernel_size=[9,9], primary_capsule_strides=[2,2], 
                 m_plus = 0.9, m_minus = 0.1, lambda_ = 0.5, alpha = 0.0005, dense1_units = 512, dense2_units=1024, img_shape=(28,28,1),
                 recon_size = None, n_qubits_pqc = 4, n_pqc_layer = 3, hybrid=True, pqc_layer="strongly", imprimitive=None, 
                 q_embedding="amplitude", noise_prop=0.0, readout_prob=0.0):
        super(HybridQuantumCapsuleNetwork, self).__init__()
        self.num_conv_layer_kernels = num_conv_layer_kernels
        self.num_primary_capsules = num_primary_capsules
        self.num_digitcaps = num_digitcaps
        self.num_primary_capsule_dim = num_primary_capsule_dim
        self.num_digitcaps_dim = num_digitcaps_dim
        self.r = r
        
        self.conv_kernel_size = conv_kernel_size
        self.conv_strides = conv_strides
        self.primary_capsule_kernel_size = primary_capsule_kernel_size
        self.primary_capsule_strides = primary_capsule_strides

        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_ = lambda_
        self.alpha = alpha

        self.reconstruction = reconstruction
        self.hybrid = hybrid
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.recon_size = img_shape
        if (not (recon_size is None)):
            self.recon_size = recon_size
        
        self.dense1_units = dense1_units
        self.dense2_units = dense2_units
        self.dense3_units = self.recon_size[0]*self.recon_size[1]*self.recon_size[2]

        self.n_qubits_pqc = n_qubits_pqc
        self.q_embedding = q_embedding
        self.QC_pqc = QuantumClass(self.n_qubits_pqc, n_pqc_layer, pqc_layer, q_embedding, imprimitive)
          
        num_caps_vec = np.floor((img_shape[-3]-self.conv_kernel_size[0])/self.conv_strides[0] + 1) # convolutional layer output
        num_caps_vec = np.floor((num_caps_vec-self.primary_capsule_kernel_size[0])/self.primary_capsule_strides[0] + 1) # primary capsule layer output
        self.num_caps_vec = np.int32(num_caps_vec*num_caps_vec*self.num_primary_capsules)
        self.caps_dim = self.num_primary_capsule_dim
        self.model_layers = []
        
        with tf.name_scope("Layers") as scope:
            self.convolution = tf.keras.layers.Conv2D(self.num_conv_layer_kernels, 
                                                      kernel_size=self.conv_kernel_size, strides=self.conv_strides, 
                                                      activation='relu', name='ConvolutionLayer')
            
            self.primary_capsule = tf.keras.layers.Conv2D(self.num_primary_capsules * self.num_primary_capsule_dim, 
                                                          kernel_size=self.primary_capsule_kernel_size, strides=self.primary_capsule_strides, 
                                                          name="PrimaryCapsule")
            self.model_layers.append(self.convolution)
            self.model_layers.append(self.primary_capsule)
                
            if self.hybrid:
                self.PQC_Layer = QuantumLayer(self.QC_pqc.qnode_pqc, self.QC_pqc.params_shape, self.num_digitcaps, noise_prop, readout_prob, name='PQC')
                self.caps_dim = 2**self.n_qubits_pqc
                if not self.q_embedding=="amplitude":
                    self.caps_dim = self.n_qubits_pqc

            if self.reconstruction:
                self.dense_1 = tf.keras.layers.Dense(units = self.dense1_units, activation='relu', name="Dense_"+str(self.dense1_units))
                self.dense_2 = tf.keras.layers.Dense(units = self.dense2_units, activation='relu', name="Dense_"+str(self.dense2_units))
                self.dense_3 = tf.keras.layers.Dense(units = self.dense3_units, activation='sigmoid', dtype='float32', name="Dense_"+str(self.dense3_units))

        if not self.hybrid:
            self.w = tf.Variable(tf.random_normal_initializer(stddev=0.01, seed=0)(
            shape=[1, self.num_caps_vec, self.num_digitcaps, self.num_digitcaps_dim, self.num_primary_capsule_dim]), 
                dtype=tf.float32, name="Weight_PrimCaps_to_DigitCaps", trainable=True)
    
    @tf.function
    def call(self, inputs): 
        x, y = inputs 
        v = self.predict_capsule_output(x) 
        reconstructed_image = None
        if self.reconstruction:        
            reconstructed_image = self.predict_recons_output(v, y)

        return v, reconstructed_image

    @tf.function
    @tf.autograph.experimental.do_not_convert
    def predict_capsule_output(self, x):
        for layer in self.model_layers:
            x = layer(x) 
        u = tf.reshape(x, (x.shape[0], -1, self.caps_dim))
        
        with tf.name_scope("CapsuleFormation") as scope:
            if self.hybrid:
                u_hat = self.PQC_Layer(u) 
                u_hat = tf.cast(u_hat, dtype=tf.float32)
            else:
                u = tf.expand_dims(u, axis=-2)
                u = tf.expand_dims(u, axis=-1)
    
                u_hat = tf.matmul(self.w, u)
                u_hat = tf.squeeze(u_hat, [-1])
        
        with tf.name_scope("DynamicRouting") as scope:
            b = tf.zeros_like(u_hat[..., :1])
            for i in range(self.r):
                c = tf.nn.softmax(b, axis=-2)
                s = tf.reduce_sum(tf.multiply(c, u_hat), axis=1, keepdims=True)
                v = self.squash(s)

                u_hat_ = tf.expand_dims(u_hat, axis=-1)
                v_ = tf.expand_dims(v, axis=-1)
                b_agreement = tf.matmul(u_hat_, v_, transpose_a=True)
                b_agreement = tf.squeeze(b_agreement, [-1])
                    
                b += b_agreement
                
        return v
    
    @tf.function
    def predict_recons_output(self, v, y):
        with tf.name_scope("Masking") as scope:
            y = tf.expand_dims(y, axis=-1)
            y = tf.expand_dims(y, axis=1)
            mask = tf.cast(y, dtype=tf.float32)
            v_masked = tf.multiply(mask, v)
            
        reconstructed_image = self.regenerate_image(v_masked)
        return reconstructed_image
        
    @tf.function
    def regenerate_image(self, inputs):
        with tf.name_scope("Reconstruction") as scope:
            v_ = tf.reshape(inputs, [-1, inputs.shape[-2] * inputs.shape[-1]])
            reconstructed_image = self.dense_1(v_)
            reconstructed_image = self.dense_2(reconstructed_image)
            reconstructed_image = self.dense_3(reconstructed_image)
        return reconstructed_image
    
    def safe_norm(self, v, epsilon=1e-7):
        v_ = tf.reduce_sum(tf.square(v), axis=-1, keepdims=True)
        return tf.sqrt(v_ + epsilon)

    def squash(self, s):
        with tf.name_scope("SquashFunction") as scope:
            s_norm = self.safe_norm(s)
        return (tf.square(s_norm)/(1 + tf.square(s_norm))) * (s/s_norm)

    def loss_function(self, v, reconstructed_image, y, x_image): 
        prediction = self.safe_norm(v)
        prediction = tf.reshape(prediction, [-1, self.num_digitcaps])
        left_margin = tf.square(tf.maximum(0.0, self.m_plus - prediction))
        right_margin = tf.square(tf.maximum(0.0, prediction - self.m_minus))
        
        l = tf.add(y * left_margin, self.lambda_ * (1.0 - y) * right_margin)
        margin_loss = tf.reduce_mean(tf.reduce_sum(l, axis=-1))
        
        loss = margin_loss
        if self.reconstruction:     
            if (not (self.recon_size is None)):
                x_image = tf.image.resize(x_image, [self.recon_size[0], self.recon_size[1]])
            if (self.recon_size[2]==1 and x_image.shape[-1]!=1):
                x_image = tf.image.rgb_to_grayscale(x_image)
                
            x_image_flat = tf.reshape(x_image, [-1, self.dense3_units])
            reconstruction_loss = tf.reduce_mean(tf.square(x_image_flat - reconstructed_image))  
            loss = tf.add(margin_loss, self.alpha * reconstruction_loss)
        return loss

    @tf.function
    def train_model(self, x, y):
        with tf.GradientTape() as tape:
            loss = self.get_loss(x, y)
            grad = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return loss

    def get_loss(self, x, y):
        y_one_hot = tf.one_hot(y, depth=self.num_digitcaps)
        v, reconstructed_image = self.call([x, y_one_hot])
        loss = self.loss_function(v, reconstructed_image, y_one_hot, x)
        return loss

    def predict_result(self, x):
        pred = self.safe_norm(self.predict_capsule_output(x))
        pred = tf.squeeze(pred, [1]) 
        argmax_pred = np.argmax(pred, axis=1)
        return argmax_pred[:,0] 

    def fit_custom(self, train_dataset, epochs=3, val_dataset=None):
        
        training_dataset_size = sum(list(map(lambda x: x[0].shape[0],list(train_dataset))))
        num_batch = len(train_dataset)

        if val_dataset is not None:
            val_dataset_size = sum(list(map(lambda x: x[0].shape[0],list(val_dataset))))

        losses = []
        accuracy = []
        valid_losses = []
        valid_accuracy = []
        for i in range(1, epochs+1):
            with tqdm(total=num_batch) as pbar:
                description = "Epoch " + str(i) + "/" + str(epochs)
                pbar.set_description_str(description)

                loss = 0
                for x_batch, y_batch in train_dataset:
                    loss += self.train_model(x_batch, y_batch)
                    pbar.update(1)

                print_statement = "Evaluating Loss ..."
                pbar.set_postfix_str(print_statement)

                loss /= num_batch
                losses.append(loss.numpy())

                print_statement = "Loss: " + str(round(loss.numpy(),6)) + " Evaluating Accuracy ..."
                pbar.set_postfix_str(print_statement)

                training_sum = sum(list(map(lambda xy: sum(self.predict_result(xy[0])==xy[1].numpy()), train_dataset)))
                acc = training_sum/training_dataset_size
                accuracy.append(acc)

                print_statement = "Loss: " + str(round(loss.numpy(),6)) + " Accuracy: " + str(round(acc,6))

                if val_dataset is not None:
                    print_statement = ("Loss: " + str(round(loss.numpy(),6)) + " Accuracy: " + str(round(acc,6))
                                       +", Evaluating Val_Loss ...")
                    pbar.set_postfix_str(print_statement)

                    val_loss = tf.reduce_mean(list(map(lambda xy: self.get_loss(xy[0], xy[1]), val_dataset)))
                    valid_losses.append(val_loss.numpy())

                    print_statement = ("Loss: " + str(round(loss.numpy(),6)) + " Accuracy: " + str(round(acc,6))
                                       +", Val_Loss: " + str(round(val_loss.numpy(),6)) + " Evaluating Val_Accuracy ...")
                    pbar.set_postfix_str(print_statement)

                    val_sum = sum(list(map(lambda xy: sum(self.predict_result(xy[0])==xy[1].numpy()), val_dataset)))
                    val_acc = val_sum/val_dataset_size
                    valid_accuracy.append(val_acc)

                    print_statement = ("Loss: " + str(round(loss.numpy(),10)) + " Acc: " + str(round(acc,6))
                                       +", Val_Loss: " + str(round(val_loss.numpy(),6)) + " Val_Acc: " + str(round(val_acc,6)))

                pbar.set_postfix_str(print_statement)

        history = {"loss":losses, "accuracy": accuracy}
        if val_dataset is not None:
            history = {"loss":losses, "accuracy": accuracy, "val_loss": valid_losses, "val_accuracy": valid_accuracy}


        return history