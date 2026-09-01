import hopsworks

project = hopsworks.login()

fs = project.get_feature_store()

print("Connected Project:", project.name)
print("Feature Store:", fs.name)
print("Hopsworks connection successful!")