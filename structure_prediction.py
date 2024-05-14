import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import utils
from tensorflow import keras

class ProteinStructurePredictor(keras.Model):
    def __init__(self, num_outputs=256):
        super().__init__()
        
        self.attention = tf.keras.layers.Attention()
        # self.attention = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=2)
        
        self.num_outputs = num_outputs
        self.flatten = keras.layers.Flatten()
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
        self.dropout = keras.layers.Dropout(0.5)

    
        # self.batch_norm1 = keras.layers.BatchNormalization()
        # self.dense1 = keras.layers.Dense(1024, activation=None)  
        # self.activation1 = keras.layers.Activation('relu')  
        # self.batch_norm2 = keras.layers.BatchNormalization()
        # self.dense2 = keras.layers.Dense(256*3)

        self.dense1 = keras.layers.Dense(1024, activation='relu')   
        self.dense2 = keras.layers.Dense(256*3)
        self.dense3 = keras.layers.Dense(256*3)

    def call(self, inputs, mask=None, inputs_primary=None, msa=None, extra_structure=None, extra_mask=None, msa_score=None):
    
        #MultiHeadAttention
        #mask_expanded_dim = tf.expand_dims(mask, axis=-1)
        #mask_tiled = tf.tile(mask_expanded_dim, [1, 1, 256])
        #x = self.attention(inputs, inputs, inputs, mask_tiled, False, False, False)
        
        x = self.attention([inputs, inputs], [tf.dtypes.cast(mask, tf.bool), tf.dtypes.cast(mask, tf.bool)])
        
        inputs_primary = tf.expand_dims(inputs_primary, axis=2) #(B,256,1)
        inputs_primary = tf.tile(inputs_primary, [1, 1, 3]) #(B,256,3)
        extra_structure = tf.transpose(extra_structure, perm=[1, 0, 2, 3])#(10,B,256,3)
        extra_mask = tf.transpose(extra_mask, perm=[1, 0, 2])#(10,B,256)
        msa = tf.expand_dims(msa, axis=3)#(B,10,256,1)
        msa = tf.tile(msa, [1, 1, 1, 3]) #(B,10,256,3)
        msa = tf.transpose(msa, perm=[1, 0, 2, 3])#(10,B,256,3)   
        msa = tf.dtypes.cast(msa, tf.float32)
        msa_score = tf.transpose(msa_score, perm=[1, 0])#(10,B)
        
        for i in range(utils.NUM_EXTRA_SEQ):
            current_msa = msa[i]
            #print(current_msa.shape)
            current_extra_structure = extra_structure[i]
            #print(current_extra_structure.shape)
            current_extra_mask = extra_mask[i]
            
            current_msa_score = msa_score[i]
            #print(current_extra_mask.shape)
            y = self.attention([tf.dtypes.cast(inputs_primary,tf.float32), tf.dtypes.cast(current_extra_structure,tf.float32), tf.dtypes.cast(current_msa,tf.float32)], 
                            [tf.dtypes.cast(mask, tf.bool), tf.dtypes.cast(current_extra_mask, tf.bool)])
            #print(x.shape, y.shape)
            
            y = y * (1 - current_msa_score)[:, np.newaxis, np.newaxis]
            
            y = self.flatten(y)         
            y = self.dense3(y)           
            y = tf.reshape(y, [-1, 256, 3])
            x = tf.concat([x, y], axis=2)      

        x = self.flatten(x)
        x = self.dense1(x)
        x= self.dropout(x) 
        x = self.dense2(x)
        
        # x = self.batch_norm1(x)
        # x = self.dense1(x)
        # x = self.activation1(x)
        # x = self.batch_norm2(x)
        # x = self.dense2(x)
        
        return tf.reshape(x, [-1, self.num_outputs, utils.NUM_DIMENSIONS]) # take batch, outputs -> batch, outputs/3, 3

def get_input_output_masks(batch):
    inputs = batch['primary_onehot']
    outputs = batch['tertiary']
    masks = batch['mask']
    msa_score = batch['msa_score']
    
    inputs_primary = batch['primary']
    msa = batch['msa']
    extra_structure = batch['extra_structure']
    extra_mask = batch['extra_mask']
    

    return inputs, outputs, masks, msa_score, inputs_primary, msa, extra_structure, extra_mask

def train(model, train_dataset, validate_dataset=None):
    total_loss = 0
    total_lddt = 0
    step = 0
    def validate():
        if validate_dataset is not None:
            for batch in validate_dataset.batch(1024):
                validate_inputs, validate_outputs, validate_masks, validate_msa_score, validate_inputs_primary, validate_msa, validate_extra_structure, validate_extra_mask = get_input_output_masks(batch)
                validate_preds = model.call(validate_inputs, validate_masks, validate_inputs_primary, validate_msa, validate_extra_structure, validate_extra_mask, validate_msa_score)
                l = utils.lddt(validate_preds, validate_outputs, validate_masks)
                validate_lddt = tf.reduce_sum(l) / validate_inputs.shape[0]
        else:
            validate_lddt = float('NaN')
        return validate_lddt
    

    for batch in train_dataset:
        inputs, labels, masks, msa_score, inputs_primary, msa, extra_structure, extra_mask = get_input_output_masks(batch)

        # with tf.GradientTape() as tape:
        #     outputs = model(inputs, masks, inputs_primary, msa, extra_structure, extra_mask)
        #     lddt_per_residue = utils.lddt(outputs, labels, masks, per_residue=True, differentiable=True)

        #     # Expand msa_score from [1024,10] to [1024,256]
        #     msa_score = tf.tile(tf.reduce_mean(msa_score, axis=1, keepdims=True), [1, 256])
        #     # Weighted lDDT using msa_score. Lower msa_score is better,use (1-msa_score) as weights.
        #     lddt = tf.reduce_sum(lddt_per_residue * (1 - msa_score) * masks) / tf.reduce_sum(masks)   
        #     loss = 1 - lddt
        
        # with tf.GradientTape() as tape:
        #     outputs = model(inputs, masks, inputs_primary, msa, extra_structure, extra_mask)
        #     lddt_per_residue = utils.lddt(outputs, labels, masks, per_residue=True, differentiable=True)

            # # Expand msa_score from [1024,10] to [1024,256]
            # msa_score = tf.tile(tf.reduce_sum(msa_score, axis=1, keepdims=True), [1, 256])
            # # Weighted lDDT using msa_score. Lower msa_score is better, use (1-msa_score) as weights.
            # lddt = tf.reduce_sum(lddt_per_residue * (1 - msa_score) * masks) / tf.reduce_sum(masks)

            # # Compute the MSE loss
            # mse_loss = tf.keras.losses.MeanSquaredError()(outputs, labels)

            # loss =  (1 - lddt) +  mse_loss
            
        with tf.GradientTape() as tape:
            outputs = model(inputs, masks, inputs_primary, msa, extra_structure, extra_mask, msa_score)
            lddt = tf.reduce_mean(utils.lddt(outputs, labels, masks, per_residue=True, differentiable=True))
            loss = 1 - lddt
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        total_lddt += lddt
        total_loss += loss
        step += 1

        #print(f'train: loss {loss:.3f} lDDT {lddt:.3f} validate: lDDT {validate():.3f}')
    print(f'train average: loss {total_loss / step:.3f} lDDT {total_lddt / step:.3f} validate: lDDT {validate():.3f}')


def test(model, test_records):
    """
    """
    for batch in test_records.batch(1024):
        # print(">"+batch[0]['id'][0])
        # print("".join([utils.AMINO_ACID_BY_INDEX[x] for x in tf.squeeze(batch[1]['primary'][0]).numpy()]))

        test_inputs, test_outputs, test_masks, test_msa_score, test_inputs_primary, test_msa, test_extra_structure, test_extra_mask = get_input_output_masks(batch)
        test_preds = model.call(test_inputs, test_masks, test_inputs_primary, test_msa, test_extra_structure, test_extra_mask, test_msa_score)
        l = utils.lddt(test_preds, test_outputs, test_masks)
        # l = utils.rmse(validate_preds, validate_outputs, validate_masks)
        test_lddt = tf.reduce_sum(l) / test_inputs.shape[0]

    print(f'test lDDT {test_lddt:.3f}')

def main(data_folder):
    training_records = utils.load_data_with_msa(data_folder, 'training_extra_msa_50_test.tfr')
    validate_records = utils.load_data_with_msa(data_folder, 'validation_extra_msa_test.tfr')
    test_records = utils.load_data_with_msa(data_folder, 'test_extra_msa_test.tfr')

    model = ProteinStructurePredictor()
    epochs = 10

    # Iterate over epochs.
    for epoch in range(epochs):
        epoch_training_records = training_records.shuffle(buffer_size=256).batch(256, drop_remainder=False)
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        train(model, epoch_training_records, validate_records)

        # test after each epoch
        test(model, test_records)

    #model.save(data_folder + '/model')


if __name__ == '__main__':
    local_home = os.path.expanduser("~")  # on my system this is /Users/jat171
    data_folder = local_home + '/Desktop/DL/new_data/'
    print(data_folder)

    main(data_folder)