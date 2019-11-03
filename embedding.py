import config
import models
import tensorflow as tf
import numpy as np

con = config.Config()
#Input training files from benchmarks/FB15K/ folder.
con.set_in_path("./project/")
#True: Input test files from the same folder.
con.set_test_triple_classification(True)
con.set_test_link_prediction(True)

con.set_work_threads(4)
con.set_train_times(500)
con.set_nbatches(100)
con.set_alpha(0.001)
con.set_margin(1.0)
con.set_bern(0)
con.set_dimension(500)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("SGD")

#Models will be exported via tf.Saver() automatically.
con.set_export_files("./project/model.vec.tf", 0)
#Model parameters will be exported to json files automatically.
con.set_out_files("./project/embedding.vec.json")
#Initialize experimental settings.
con.init()
#Set the knowledge embedding model
con.set_model(models.TransE)
#Train the model.
con.run()

#Get the embeddings (numpy.array)
embeddings = con.get_parameters("numpy")
#To test link prediction after training needs "set_test_link_prediction True)".
#To test triple classfication after training needs "set_test_triple_classification(True)"
con.test()
