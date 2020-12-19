import tensorflow as tf
import dataset as ds
import numpy as np

'''
 * @author Nassim MOUALEK
 * @since 06/12/2020
'''

if __name__ == '__main__':
    rows = 200
    features, targets = ds.generateDataset(rows)
    features = ds.extendDataset(features)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(36, activation='tanh'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(12, activation='tanh'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(2, activation=None)
    ])

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    model.fit(features, targets, epochs=300)

    features2, targets2 = ds.generateDataset(rows) #generate test data
    features2 = ds.extendDataset(features2)

    model.evaluate(features2, targets2, verbose=2)
    toPredict = ds.extendDataset(np.array([[-2, -2], [2, -2]]))
    predicted = model.predict(toPredict)
    print("predicted", predicted)
    print("predicted", tf.math.softmax(predicted) )
    predicted = tf.argmax(input=predicted, axis=1).numpy()
    for i in range(len(toPredict)):
        print("X=%s, Predicted=Patient is %s" % (toPredict[i], np.where(predicted[i], "healthy", "sick")))

    ds.show(features, targets)
