import tensorflow as tf

import dataset as ds

'''
 * @author Nassim MOUALEK
 * @since 06/12/2020
'''

if __name__ == '__main__':
    rows = 200
    features, targets = ds.getSprialDataset(rows)
    features = ds.extendDataset(features)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(36, activation='tanh'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(12, activation='tanh'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(4, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(features, targets, epochs=300)
    features2, targets2 = ds.getSprialDataset(rows)
    features2 = ds.extendDataset(features2)

    model.evaluate(features2, targets2, verbose=2)

    ds.show(features, targets)
