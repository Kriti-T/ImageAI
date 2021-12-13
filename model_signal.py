from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="signal")
trainer.setTrainConfig(object_names_array=['signal'], batch_size=64, num_experiments=20)
trainer.trainModel()