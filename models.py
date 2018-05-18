import sys
from helpers import *
from tensorflow.contrib.layers import flatten

def lr_model(k, c, d, kmeans=None):
    """Sklearn logistic regressionmodel for classifying images based on image features 
    detected with SIFT algorithm and kmeans algorithm."""
    
    train_features, train_classes, train_transformed_image_names = read_transform_all_images(d)
    print('Creating kmeans for feature detection.')
    if(not kmeans):
        kmeans = create_cluster(np.concatenate(train_features), k)
        joblib.dump(kmeans, os.path.join(path_to_root, 'models/aux/kmeans.sav'))

    train_samples = get_train_samples(train_features, kmeans, 'model1')
    lr = LogisticRegression(penalty='l2', C=c)
    lr.fit(train_samples, train_classes)
    train_score = lr.score(train_samples, train_classes)
    print("Training accuracy: %s " %train_score)
    #Save the model
    filename = os.path.join(path_to_root, 'models/%s/saved/model.sav' % ('model1'))
    joblib.dump(lr, filename)
    return train_score

def lr_model_test(d, m='model1'):
    """Loads both the kmeans model and the linear regression and predict the labels of the images contained in d directory"""
    test_features, test_classes, test_transformed_image_names = read_transform_all_images(d, False)
    try:
        kmeans = joblib.load(os.path.join(path_to_root, 'models/aux/kmeans.sav'))
    except Exception as e:
        print("Could not load Kmeans model. Train first or verify model's name.")
        print(e)
        return
    try:
        lr = joblib.load(os.path.join(path_to_root, 'models/%s/saved/model.sav' % (m)))
    except:
        print("Could not load sklearn logistic regression model. Train first or verify model's name.")
        return
    test_samples = get_test_samples(test_features, kmeans, m)

    test_score = lr.score(test_samples, test_classes)
    print("Testing accuracy: %s" % test_score)
    return test_score, lr.predict(test_samples)


def tf_lr_model(k, d, learning_rate = 0.001, training_iteration = 150, batch_size = 50, early_stopping = 0.001, display_step = 2, kmeans=None):
    """Tensor flow softmax model for classifying images based on image features detected with SIFT algorithm and kmeans algorithm."""
    # Set parameters
    train_features, train_classes, train_transformed_image_names = read_transform_all_images(d)
    print('Creating kmeans for feature detection.')

    if(not kmeans):
        kmeans = create_cluster(np.concatenate(train_features), k)
        joblib.dump(kmeans, os.path.join(path_to_root, 'models/aux/kmeans_tf.sav'))

    features = get_train_samples(train_features, kmeans, 'model2').values
    enc = OneHotEncoder(43)
    labels = enc.fit_transform(train_classes.reshape(-1, 1)).todense()
    n_samples, n_features = features.shape
    _, n_classes = labels.shape

    tf.reset_default_graph()

    # TF graph input
    x = tf.placeholder("float", [None, n_features]) # Based on Kmeans, there are inputs are of shape k
    y = tf.placeholder("float", [None, n_classes]) # 42 classes.

    # Set model weights
    W = tf.Variable(tf.zeros([n_features, n_classes]))
    b = tf.Variable(tf.zeros([n_classes]))

    # Construct a linear model
    model = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

    # Minimize error using log loss
    cost_function = tf.losses.log_loss(y, model)

    # Optimizer, AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        for iteration in range(training_iteration):
            avg_cost = 0.
            total_batch = int(len(features)/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = random_batch(features, labels, batch_size)
                # Fit training using batch data
                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
                # Compute average loss
                avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})/total_batch
                #TODO Implement early stopping, I just got the # of iterations with hypterparameter tunning when I wanted to stop
                    
            # Display logs per eiteration step
            if iteration % display_step == 0:
                print ("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))

        print("Tuning completed!")

        # Test the model
        predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
        # Save the model
        saver = tf.train.Saver()
        save_path = saver.save(sess, os.path.join(path_to_root, "models/model2/saved/model2.ckpt"))
        
        # Calculate accuracy on training set
        accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
        acc = accuracy.eval({x: features, y: labels})
        print("Training accuracy:" , acc)
        return acc

def tf_lr_test(d, m='model2'):
    """Loads both the kmeans model and the softmax model and predict the labels of the images contained in d directory"""
  
    #Load features from images and kmeans model
    test_features, test_classes, test_transformed_image_names = read_transform_all_images(d, False)
    try:
        kmeans = joblib.load(os.path.join(path_to_root, 'models/aux/kmeans_tf.sav'))
    except:
        print("Could not load Kmeans model. Train first or verify model's name.")
        return

    features = get_test_samples(test_features, kmeans, m).values
    n_samples, n_features = features.shape
    #Encode the labels, TODO: get #classes.
    enc = OneHotEncoder(43)
    labels = enc.fit_transform(test_classes.reshape(-1, 1)).todense()
    _, n_classes = labels.shape

    tf.reset_default_graph()

    # TF input
    x = tf.placeholder("float", [None, n_features]) # Based on Kmeans, there are inputs are of shape k
    y = tf.placeholder("float", [None, n_classes]) # 43 classes.

    # Define variables to restore
    W = tf.Variable(tf.zeros([n_features, n_classes]))
    b = tf.Variable(tf.zeros([n_classes]))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        model = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
        predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
        #Calculate correct predictions, get mean value for accuracy.
        accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
        try:
            #Restore the model
            saver.restore(sess, os.path.join(path_to_root, "models/%s/saved/%s.ckpt" %(m, m)))
            #Eval accuracy
            acc = accuracy.eval({x: features, y: labels})
            print ("Testing accuracy:", acc)
            return acc, tf.argmax(model, 1).eval({x: features})
        except Exception as e: 
            print('Error using %s. Train first.' %m)
            print(e)
            return

def LeNet_5(x):
    """LeNet architecture. I used tanh non-linearity as the paper states. I made a slight modification to the input
    of the model to be able to detect features in rgb channels rather than in a grayscale image. This is because many
    traffic signs are similar in shape but not in color.
    """

    #Weights and biases for conv layer #1 shape from 32x32x3 to 28x28x6 and pooled to 14x14x6
    conv1_w = tf.Variable(tf.truncated_normal(shape = [5,5,3,6],mean = 0, stddev = 0.1), name='conv1_w')
    conv1_b = tf.Variable(tf.zeros(6), name='conv1_b')
    conv1 = tf.nn.conv2d(x,conv1_w, strides = [1,1,1,1], padding = 'VALID') + conv1_b 
    conv1 = tf.nn.tanh(conv1)
    #Average pooling for layer #1
    pool_1 = tf.nn.avg_pool(conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID', name='pool_1')

    #Weights and biases for conv layer #2 shape from 14x14x6 to 10x10x16 and pooled to 5x5x16 
    conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = 0, stddev = 0.1), name='conv2_w')
    conv2_b = tf.Variable(tf.zeros(16), name='conv2_b')
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b
    conv2 = tf.nn.tanh(conv2)
    #Average pooling for layer #2
    pool_2 = tf.nn.avg_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID', name='pool_2')

    #Flatten to fully connect with layer 3 shape from 5x5x16 to 400x1
    fc1 = flatten(pool_2)

    #Weights and biases for layer #3 shape from 400x1 to 120x1
    fc1_w = tf.Variable(tf.truncated_normal(shape = (400,120), mean = 0, stddev = 0.1), name='fc1_w')
    fc1_b = tf.Variable(tf.zeros(120), name='fc1_b')
    fc1 = tf.matmul(fc1,fc1_w) + fc1_b
    fc1 = tf.nn.tanh(fc1)

    #Weights and biases for layer #4 shape from 120x1 to 84x1
    fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = 0, stddev = 0.1), name='fc2_w')
    fc2_b = tf.Variable(tf.zeros(84), name='fc2_b')
    fc2 = tf.matmul(fc1,fc2_w) + fc2_b
    fc2 = tf.nn.tanh(fc2)

    #Weights and biases for layer #5 shape from 84x1 to 43x1
    fc3_w = tf.Variable(tf.truncated_normal(shape = (84,43), mean = 0 , stddev = 0.1), name='fc3_w')
    fc3_b = tf.Variable(tf.zeros(43), name='fc3_b')
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    return logits #return the output of layer 5 to be computed with a softmax activation function.

def tf_lenet_model(x_data, y_data, mode='train', lr=0.001, n_epochs=100, batch_size=60, verbose=True):
    """Wrapper for the model and also applies softmax to get the logits. Trains, tests and use saved model in training for inference."""
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, shape=[None,32,32,3])
    y_ = tf.placeholder(tf.float32, shape=[None, 43])
    logits = LeNet_5(x) #TF model, defines the graph.
    y = tf.nn.softmax(logits) #Softmax output
    
    #Save variables to be restored in the future
    var_save = [v for v in tf.trainable_variables()]
    saver = tf.train.Saver(var_save)
    with tf.Session() as sess:
        if(mode == 'train'):
            # Minimize the mean squared errors. 
            loss = tf.reduce_mean(tf.square(tf.subtract(y_, y)))
            #It is supposed to be stochastic gradient descent, AdamOptimizer is similar.
            optimizer = tf.train.AdamOptimizer(learning_rate = lr)
            train = optimizer.minimize(loss)
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            init = tf.global_variables_initializer()
            sess.run(init)
            
            #Training iterations
            for epoch in range(n_epochs):
                avg_cost = 0.
                total_batch = int(len(x_data)/batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_xs, batch_ys = random_batch(x_data, y_data, batch_size)
                    # Fit training using batch data
                    sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
                    # Compute average loss
                    avg_cost += sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})/total_batch
                    # TODO Implement early stopping, I just got the # of iterations with hypterparameter tunning when I wanted to stop

                # Display logs per eiteration step
                if ((epoch % 5 == 0) and verbose):
                    print ("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

            accuracy = sess.run(accuracy_operation, feed_dict={x: x_data, y_: y_data})
            print('Training accuracy: ', accuracy)
            save_path = saver.save(sess, os.path.join(path_to_root, "models/model3/saved/model3.ckpt"))
            return accuracy

        elif(mode == 'test'):
            #Load an existant model and check accuracy
            try: 
                saver.restore(sess, os.path.join(path_to_root, "models/model3/saved/model3.ckpt"))
            except:
                print('Error loading model3, try running training first.')
                return
            predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_data, 1))
            accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
            acc = accuracy.eval(feed_dict={x:x_data})
            print('Testing accuracy: ', acc)
            return acc

        elif(mode == 'infer'):
            try: 
                saver.restore(sess, os.path.join(path_to_root, "models/model3/saved/model3.ckpt"))
            except:
                print('Error loading model3, try running training first.')
                return

            predictions = tf.argmax(y, 1)
            return predictions.eval(feed_dict={x:x_data})