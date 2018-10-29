import tensorflow as tf
import numpy as np
import argparse 
import sys 
import random
import time
# from tensorflow import slim 
import os
from tensorflow.python import debug as tf_debug
import h5py 
from scipy.sparse import load_npz
import sys 

###### loading data ############


data_train=load_npz('data_train.npz');
data_train=data_train.toarray();

data_test=load_npz('data_test.npz');
data_test=data_test.toarray();

y_train=load_npz('y_train.npz');
y_test=load_npz('y_test.npz');
y_train=y_train.toarray();
y_test=y_test.toarray();



stale_parameter=int(sys.argv[1][-2:]);
print('staleness factor chosen is {}'.format(stale_parameter));

############### accuracy prediction function ##################

def accuracy_eval(y_test,y_pred):
    s=0;
    
    y_pred = np.argmax(y_pred,axis=1);
    
    for i in range(len(y_test)):
        if y_test[i,y_pred[i]]>0:
            s+=1;
    return s/len(y_test);







##### Servers inititalisation ##############

parameter_servers=["10.1.1.254:2225"];
workers=["10.1.1.253:2223","10.1.1.252:2224"];
cluster = tf.train.ClusterSpec({"ps":parameter_servers,"worker":workers});


##### input flags ################

tf.app.flags.DEFINE_string("job_name","","'ps' / 'worker'");
tf.app.flags.DEFINE_integer("task_index",0,"Index of task within the job");
FLAGS=tf.app.flags.FLAGS;

### setup ########
config=tf.ConfigProto();
config.gpu_options.allow_growth=True;
config.allow_soft_placement=True;
config.log_device_placement=True;

server=tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index,config=config)




## logs##
log_dir="./logs/ssp_"+str(stale_parameter)+'/';


if FLAGS.job_name=='ps':
    server.join();
elif FLAGS.job_name=='worker':
    ################################################################### between-graph replication  ###############################################################################
    with tf.device(tf.train.replica_device_setter(
    worker_device="/job:worker/task:%d" % FLAGS.task_index,
    cluster=cluster)):
        
        #### Hyperparameters #################

        batch_size=7000;

        num_examples = len(data_train);

        num_batches_per_epoch = int(num_examples/batch_size);

        num_epochs=100;

        reg_constant=.001



        initializer = tf.contrib.layers.xavier_initializer()
        regularizer = tf.contrib.layers.l2_regularizer(scale = reg_constant)

        
        
        
        global_step = tf.get_variable('global_step', [], 
                                initializer = tf.constant_initializer(0), 
                                trainable = False,dtype=tf.int32)
    
        starter_learning_rate = 0.005

#         learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               # decay_steps=1, decay_rate=0, staircase=True)

        inp =tf.placeholder(shape=(None,data_train.shape[1]),dtype=tf.float64);

        labels=tf.placeholder(shape=(None,50),dtype=tf.float64);

        W=tf.get_variable('W',shape=(data_train.shape[1],50),dtype=tf.float64,\
                                                     initializer=initializer, regularizer=regularizer);

        b=tf.get_variable('b',shape=(50,),dtype=tf.float64,\
                         regularizer=regularizer );

        out=tf.add(tf.matmul(inp,W),b);

        out_logits=tf.nn.softmax(out);



        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES);
    #     print(out_logits.shape)
        loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=out));
        loss= loss + reg_constant*sum(reg_losses);
        optimizer=tf.train.AdamOptimizer(learning_rate=starter_learning_rate);

        
        
        replicated_opt=tf.contrib.opt.DropStaleGradientOptimizer(opt=optimizer,staleness=stale_parameter);
        
        final_optimizer = replicated_opt.minimize(loss, global_step=global_step);
        
        
        
        
        # init_token = replicated_opt.get_init_tokens_op();##Returns the op to fill the sync_token_queue with the tokens.
        
        # queue_runner = replicated_opt.get_chief_queue_runner(); #Returns the QueueRunner for the chief to execute.
        
        
    #     optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss);
        
        
        
        init_op = tf.global_variables_initializer()
        
        tf.summary.scalar("loss", loss)
#         tf.summary.scalar("Accuracy", accuracy(test_prediction,label_test))
        # merge all summaries into a single "operation" which we can execute in a session 
        
        summary_op = tf.summary.merge_all();
        
        # init_token = tf.global_variables_initializer(); ## No Need 
        
        sv_obj = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),\
                            global_step=global_step,\
                            init_op=init_op);
        
        
        
        
        
    with sv_obj.prepare_or_wait_for_session(server.target) as session:

        # create log writer object (this will log on every machine)
        writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph());

        cost_plot_task0=[];
        cost_plot_task1=[];

        # if FLAGS.task_index == 0:
        #     sv_obj.start_queue_runners(session, [queue_runner]);
        #     session.run(init_token);
        
        
        for curr_epoch in range(num_epochs):
            test_accuracy = 0
            train_cost = 0
            start = time.time()



            #iterating over total number of batchs
            for batch in range(num_batches_per_epoch):

                # Getting the index

                indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]

    #             data_train=np.zeros((batch_size,len(word_to_index)));
    #             for ind in range(len(indexes)):
    #                 for j in range(len( x_train[indexes[i]] ) ):
    #                     data_train[ind][  x_train[indexes[i]]  ]=1;


                batch_train_inputs = data_train[indexes];
                batch_train_inputs=batch_train_inputs;
                batch_train_targets=y_train[indexes];



                # feed dictionay for training
                feed = {inp: batch_train_inputs,
                        labels: batch_train_targets,}

                batch_cost, _ = session.run([loss, final_optimizer], feed)
                train_cost += batch_cost*batch_size




    #         # Shuffle the data
            # shuffled_indexes = np.random.permutation(num_examples)## shuffling data for next epoch
            # data_train = data_train[shuffled_indexes];
            # data_train = data_train[shuffled_indexes];

            # Metrics mean
            train_cost /= num_examples;

            if FLAGS.task_index==0:
                cost_plot_task0.append(train_cost);
            else:
                cost_plot_task1.append(train_cost);



            if(curr_epoch%1==0):

                test_start=time.time();

                batch_test_inputs = data_test;
                batch_test_inputs=batch_test_inputs;


                # feed dictionary for test data to calculate test CER

                feed_test = {
                            inp: batch_test_inputs,

                            }

                y_pred = session.run(out_logits,feed_test)

                test_accuracy=accuracy_eval(y_test,y_pred);

                test_end=time.time();


                log = "Epoch {}/{}, train_cost = {:.3f}, test_accuracy = {:.3f}, time = {:.3f},test-time={:.3f}"
                print(log.format(curr_epoch+1, num_epochs, train_cost, test_accuracy, time.time() - start,test_end-test_start))


            
                    
                
    sv_obj.stop();

    np.save(log_dir+'worker0.npy',cost_plot_task0);
    np.save(log_dir+'worker1.npy',cost_plot_task1);

    print('Model has been trained ');

        
        
        
        
        
        
        
        




