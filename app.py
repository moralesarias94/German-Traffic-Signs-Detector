from models import *
import click

@click.group()
def cli():
    pass

@cli.command()
@click.option('-m', default='model1', help='Model to use e.g model1')
@click.option('-d', default='images/train/',
              help='Path to directory with trainig data')
def train(m, d):
    """Receives a model name e.g model1 and a directory where training data is e.g images/train creates samples from
        the training data and trains a model with some hyperparameters already tunned.  
    """
    if(m == 'model1'):
        #Throw exception if m or d invalid
        k = 600; c = 0.1
        print('Training %s with k=%s c=%s' %(m, k, c))        
        lr_model(k, c, d)
        return 
    elif(m == 'model2'):
        k = 700
        print('Training %s with k=%s' %(m, k))  
        tf_lr_model(k, d)
    elif(m == 'model3'):
        imgs, labels, tf_img_names = get_images('images/train', infer=False)
        imgs, labels = transform_input(imgs, labels, infer=False)
        tf_lenet_model(imgs, labels, 'train', n_epochs=100)
    else:
        print('Incorrect model name. Try model1')
        
@cli.command()
@click.option('-m', default='model1', help='Model to use e.g model1')
@click.option('-d', default='images/train/',
              help='Path to directory with testing data')
def test(m, d):
    """Call this function with a model's name and a directory with data for testing. It loads the model and perform predictions 
        in the testing data and shows the model's accuracy.
    """
    if(m == 'model1'):
        #Throw exception if m or d invalid
        lr_model_test(d)
        return
    elif(m=='model2'):
        #Runs testing
        tf_lr_test(d)

    elif(m == 'model3'):
        imgs, labels, tf_img_names = get_images('images/test', infer=False)
        imgs, labels = transform_input(imgs, labels, infer=False)
        tf_lenet_model(imgs, labels, 'test')

    else:
        print('Incorrect model name. Try model1')

@cli.command()
@click.option('-m', default='model1', help='Model to use e.g model1')
@click.option('-d', default='images/user/',
              help='Path to directory with user data')
def infer(m, d):
    
    #Convert this into a function
    if(m == 'model1'):
        user_features, _, user_transformed_image_names = read_transform_all_images(d, True)
        try:
            kmeans = joblib.load(os.path.join(path_to_root,'models/aux/kmeans.sav'))
        except:
            print("Could not load Kmeans model. Train first or verify model's name.")
            return
        try:
            lr = joblib.load(os.path.join(path_to_root,'models/%s/saved/model.sav' % (m)))
        except:
            print("Could not load sklearn logistic regression model. Train first or verify model's name.")
            return
        user_samples = get_test_samples(user_features, kmeans, m)
        preds = lr.predict(user_samples)

    elif(m == 'model2'):
        user_features, _, user_transformed_image_names = read_transform_all_images('images/user', True)
        try:
            kmeans = joblib.load(os.path.join(path_to_root,'models/aux/kmeans_tf.sav'))
        except:
            print("Could not load Kmeans model. Train first or verify model's name.")
            return
        #Get samples, convert it from dataframe to numpy ndarray.
        k = len(kmeans.cluster_centers_)
        user_samples = get_test_samples(user_features, kmeans, m).values
        #Convert labels into OneHot arrays.

        tf.reset_default_graph()
        # TF graph input
        x = tf.placeholder("float", [None, k]) # Based on Kmeans, there are inputs are of shape k
        y = tf.placeholder("float", [None, 43]) # 42 classes.

        # Create a model

        # Set model weights
        W = tf.Variable(tf.zeros([k, 43]))
        b = tf.Variable(tf.zeros([43]))
        saver = tf.train.Saver()

        with tf.Session() as sess:
            model = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
            user_forward = tf.argmax(model, 1)
            try:
                #Restore the model
                saver.restore(sess, os.path.join(path_to_root,"models/model2/saved/model2.ckpt"))
                #Run predictions
                preds = sess.run(user_forward, feed_dict={x:user_samples})
            except Exception as e:
                print('Error using %s. Train first.' %m)
                print(e)
                return
    elif(m == 'model3'):
        imgs, labels, user_transformed_image_names = get_images(d, infer=True)
        imgs, labels = transform_input(imgs, labels, infer=True)
        preds = tf_lenet_model(imgs, labels, 'infer')

    show_predictions(preds, d, user_transformed_image_names)


@cli.command()
def download():
    """This function downloads the training and test images 
    from  http://benchmark.ini.rub.de/Dataset_GTSDB"""

    data_sets_urls = {'test': 'http://benchmark.ini.rub.de/Dataset_GTSDB/TestIJCNN2013.zip',
                        'train': 'http://benchmark.ini.rub.de/Dataset_GTSDB/TrainIJCNN2013.zip',
                        'full': 'http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip'}
    base_output_dir = os.path.join(path_to_root, 'images/')
    url = data_sets_urls['full']
    output_full_dir = os.path.join(base_output_dir, 'full')
    full_files = os.listdir(output_full_dir)
    full_files.remove('.gitignore')
    full_files.remove('.DS_Store')
    file_name = os.path.join(base_output_dir, "full.zip")
    if(not os.path.exists(file_name)):
        print("Downloading images from ", url, " to: ", output_full_dir)
        
        with open(file_name, "wb") as f:
                print ("Downloading %s" % file_name)
                response = requests.get(url, stream=True)
                total_length = response.headers.get('content-length')

                if total_length is None: # no content length header
                    f.write(response.content)
                else:
                    dl = 0
                    total_length = int(total_length)
                    for data in response.iter_content(chunk_size=4096):
                        dl += len(data)
                        f.write(data)
                        done = int(50 * dl / total_length)
                        sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )    
                        sys.stdout.flush()
    if(len(full_files) == 0):
        print('Unziping images into full folder.')
        zip_data_set = zipfile.ZipFile(file_name, 'r')
        for zip_info in zip_data_set.infolist():
            filename = zip_info.filename
            if((filename.count('/') > 1 and filename[-1] != '/')):
                zip_info.filename = os.path.split(os.path.dirname(filename))[1] + os.path.basename(filename)
                zip_data_set.extract(zip_info, output_full_dir)
                
    print('Realocating train and test images')
    full_files = os.listdir(output_full_dir)
    full_files.remove('.gitignore')
    full_files.remove('.DS_Store')
    all_file_names = pd.DataFrame(full_files)
    train, test = train_test_split(all_file_names, test_size=0.2)
    train_output_dir = os.path.join(base_output_dir, 'train')
    test_output_dir = os.path.join(base_output_dir, 'test')
    train_output_dir_list = os.listdir(train_output_dir)
    test_output_dir_list = os.listdir(test_output_dir)
    #TODO: Find a better solution for this
    train_output_dir_list.remove('.gitignore')
    train_output_dir_list.remove('.DS_Store')
    test_output_dir_list.remove('.gitignore')
    test_output_dir_list.remove('.DS_Store')

    if(len(train_output_dir_list) == 0 and len(test_output_dir_list) == 0):
        train[0].apply(lambda x: copyfile(os.path.join(output_full_dir, x), os.path.join(train_output_dir, x)))
        test[0].apply(lambda x: copyfile(os.path.join(output_full_dir, x), os.path.join(test_output_dir, x)))
    else:
        print('Train and test folders should be empty.')

if __name__ == '__main__':
    cli()
