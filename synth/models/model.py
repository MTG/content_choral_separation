# The template model class
import tensorflow as tf
from synth.config import config
from synth.data_pipeline import autovc as data_pipeline
import soundfile as sf
import matplotlib.pyplot as plt
import os



class Model(object):
    def __init__(self):
        self.check_prep()
        self.get_placeholders()
        self.model()


    def test_file_all(self, file_name, sess):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        scores = self.extract_f0_file(file_name, sess)
        return scores

    def validate_file(self, file_name, sess):
        """
        Function to extract multi pitch from file, for validation. Currently supports only HDF5 files.
        """
        scores = self.extract_f0_file(file_name, sess)
        pre = scores['Precision']
        acc = scores['Accuracy']
        rec = scores['Recall']
        return pre, acc, rec


    def check_prep(self):
        """
        Check if the data is prepared for the datasets to be used and then check stats
        """
        if "NUS" in config.datasets and not config.nus_prep:
            from synth.data_prep import prepare_nus
            prepare_nus.process_dataset()
        if "JVS" in config.datasets and not config.jvs_prep:
            from synth.data_prep import prepare_jvs
            prepare_jvs.process_dataset()
        if "CSD" in config.datasets and not config.csd_prep:
            from synth.data_prep import prepare_csd
            prepare_csd.process_dataset()
        if "DAMP" in config.datasets and not config.damp_prep:
            from synth.data_prep import prepare_damp
            prepare_damp.process_dataset()
        if "BACH" in config.datasets and not config.bach_prep:
            from synth.data_prep import prepare_bach
            prepare_bach.process_dataset()

        if not config.stat_prep:
            data_pipeline.get_stats()
            config.change_variable("stat_prep", "prep", "True")

    def load_model(self, sess, log_dir):
        """
        Load model parameters, for synthesis or re-starting training. 
        """
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep= config.max_models_to_keep)


        sess.run(self.init_op)

        ckpt = tf.train.get_checkpoint_state(log_dir)

        if ckpt and ckpt.model_checkpoint_path:
            print("Using the model in %s"%ckpt.model_checkpoint_path)
            self.saver.restore(sess, ckpt.model_checkpoint_path)


    def save_model(self, sess, epoch, log_dir):
        """
        Save the model.
        """
        checkpoint_file = os.path.join(log_dir, 'model.ckpt')
        self.saver.save(sess, checkpoint_file, global_step=epoch)
        print("Model saved at epoch {}".format(epoch))

    def print_summary(self, print_dict, epoch, duration):
        """
        Print training summary to console, every N epochs.
        Summary will depend on model_mode.
        """

        print('epoch %d took (%.3f sec)' % (epoch + 1, duration))
        for key, value in print_dict.items():
            print('{} : {}'.format(key, value))

    def get_summary(self, sess, log_dir, summary_dict):
        """
        Gets the summaries and summary writers for the losses.
        """
        summaries = {}

        for key in summary_dict.keys():
            summaries[key] = tf.summary.scalar(key, summary_dict[key])

        self.train_summary_writer = tf.summary.FileWriter(log_dir+'/train/', sess.graph)
        self.val_summary_writer = tf.summary.FileWriter(log_dir+'/val/', sess.graph)
        self.summary = tf.summary.merge_all()

    # def find_gender(self, singer_name):



    def plot_features(self, feat_dict):
        """
        Plots a set of features, with the ground truth and the output as in the directory.
        """

        for num, feature in enumerate(feat_dict.keys()):
            plt.figure(num)
            gt = feat_dict[feature]['gt']
            op = feat_dict[feature]['op']
            if len(gt.shape) == 1 or gt.shape[-1] == 1:
                plt.plot(gt, label = "Ground Truth {}".format(feature))
                plt.plot(op, label = "Output {}".format(feature))
                if "notes" in feat_dict[feature].keys():
                    plt.plot(feat_dict[feature]["notes"], label = "Notes")
                plt.legend()
            else:
                ax1 = plt.subplot(211)

                plt.imshow(gt.T,aspect='auto',origin='lower')

                ax1.set_title("Ground Truth {}".format(feature, fontsize=10))

                ax2 =plt.subplot(212, sharex = ax1, sharey = ax1)

                ax2.set_title("Output {}".format(feature, fontsize=10))

                plt.imshow(op.T,aspect='auto',origin='lower')

                ax2.set_title("Output {}".format(feature, fontsize=10))

        
        plt.show()
